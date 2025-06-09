/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use notify::Event;
use notify::RecursiveMode;
use notify::Watcher as _;
use notify::recommended_watcher;
use tracing::debug;
use tracing::info;
use watchman_client::CanonicalPath;
use watchman_client::Connector;
use watchman_client::Subscription;
use watchman_client::SubscriptionData;
use watchman_client::expr::Expr;
use watchman_client::pdu::Clock;
use watchman_client::pdu::FileType;
use watchman_client::pdu::SubscribeRequest;
use watchman_client::pdu::SyncTimeout;

pub struct Watcher(WatcherInner);

enum WatcherInner {
    Watchman(Watchman),
    Notify(NotifyWatcher),
}

impl Watcher {
    pub async fn wait(&mut self) -> anyhow::Result<Vec<Event>> {
        match &mut self.0 {
            WatcherInner::Watchman(w) => w.wait().await,
            WatcherInner::Notify(w) => w.wait().await,
        }
    }

    pub async fn watchman(path: &Path) -> anyhow::Result<Self> {
        Ok(Self(WatcherInner::Watchman(Watchman::new(path).await?)))
    }

    pub fn notify(paths: &[PathBuf]) -> anyhow::Result<Self> {
        Ok(Self(WatcherInner::Notify(NotifyWatcher::new(paths)?)))
    }
}

struct NotifyWatcher {
    receiver: Receiver<notify::Result<Event>>,
}

impl NotifyWatcher {
    fn new(paths: &[PathBuf]) -> anyhow::Result<Self> {
        let (sender, receiver) = channel();
        let mut watcher = recommended_watcher(sender)?;
        for path in paths {
            watcher.watch(path, RecursiveMode::Recursive)?;
        }
        Ok(Self { receiver })
    }

    async fn wait(&mut self) -> anyhow::Result<Vec<Event>> {
        let mut res = Vec::new();
        res.push(self.receiver.recv()??);
        // Wait up to 0.1s to buffer up events
        let end = Instant::now() + Duration::from_secs_f32(0.1);
        while let Some(remaining) = end.checked_duration_since(Instant::now()) {
            match self.receiver.recv_timeout(remaining) {
                Ok(event) => res.push(event?),
                Err(_) => break,
            }
        }
        Ok(res)
    }
}

// The "new" field is marked as deprecated, but buck2 uses it and
// I'm unaware of issues due to its use there.
//
// Putting this in it own mod was the best way to scope the allow(deprecated).
#[allow(deprecated)]
mod watchman_query {
    use std::path::Path;

    use notify::Event;
    use notify::EventKind;
    use notify::event::CreateKind;
    use notify::event::EventAttributes;
    use notify::event::ModifyKind;
    use notify::event::RemoveKind;
    use serde::Deserialize;
    use watchman_client::prelude::*;

    query_result_type! {
        pub struct SubscriptionFields {
            name: NameField,
            file_type: FileTypeField,
            exists: ExistsField,
            new: NewField,
        }
    }

    impl SubscriptionFields {
        pub fn into_event(self, root: &Path) -> Option<Event> {
            match *self.file_type {
                FileType::BlockSpecial
                | FileType::CharSpecial
                | FileType::Fifo
                | FileType::Socket
                | FileType::SolarisDoor
                | FileType::Unknown => {
                    return None;
                }
                FileType::Directory | FileType::Regular | FileType::Symlink => {}
            };

            let kind = match (*self.exists, *self.new) {
                (true, true) => EventKind::Create(CreateKind::Any),
                (false, _) => EventKind::Remove(RemoveKind::Any),
                (true, false) => EventKind::Modify(ModifyKind::Any),
            };

            Some(Event {
                kind,
                paths: vec![root.join(self.name.as_path())],
                attrs: EventAttributes::new(),
            })
        }
    }
}

struct Watchman {
    root: PathBuf,
    subscription: Subscription<watchman_query::SubscriptionFields>,
}

impl Watchman {
    async fn wait(&mut self) -> anyhow::Result<Vec<Event>> {
        loop {
            match self
                .subscription
                .next()
                .await
                .context("Failed to get watchman response")?
            {
                SubscriptionData::Canceled => {
                    return Err(anyhow::anyhow!("Watchman subscription was canceled"));
                }
                SubscriptionData::StateEnter { .. } | SubscriptionData::StateLeave { .. } => {
                    continue;
                }
                SubscriptionData::FilesChanged(changes) => {
                    if let Some(files) = changes.files {
                        info!("Received watchman changes on {} files", files.len());
                        return Ok(files
                            .into_iter()
                            .filter_map(|file| file.into_event(&self.root))
                            .collect());
                    } else {
                        continue;
                    }
                }
            }
        }
    }

    async fn new(path: &Path) -> anyhow::Result<Self> {
        let root = CanonicalPath::canonicalize(path)?;
        let watchman_client = Connector::new()
            .connect()
            .await
            .context("Failed to connect to watchman client")?;
        let root = watchman_client
            .resolve_root(root)
            .await
            .context("Failed to resolve root directory for watchman client")?;
        let clock = watchman_client
            .clock(&root, SyncTimeout::Default)
            .await
            .context("Failed to obtain initial clock from watchman client")?;
        let (subscription, initial_response) = watchman_client
            .subscribe::<watchman_query::SubscriptionFields>(
                &root,
                SubscribeRequest {
                    // If `empty_on_fresh_instance` is false (default), and if `since` is None,
                    // watchman is going to crawl & return the entire file tree on the first subscription
                    // response, which can be very slow. Explicitly setting the `since` clock can
                    // significantly speed up the setup.
                    since: Some(Clock::Spec(clock)),
                    relative_root: Some(PathBuf::from(".")),
                    expression: Some(Expr::All(vec![
                        Expr::FileType(FileType::Regular),
                        Expr::Suffix(vec![PathBuf::from("py"), PathBuf::from("pyi")]),
                    ])),
                    ..SubscribeRequest::default()
                },
            )
            .await
            .context("Failed to subscribe to changes")?;
        debug!(
            "Successfully established watchman subscription. Initial response: {:#?}",
            initial_response
        );
        Ok(Watchman {
            root: root.path(),
            subscription,
        })
    }
}
