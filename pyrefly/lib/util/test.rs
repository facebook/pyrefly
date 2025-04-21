/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
use std::io::Write;
use std::path::Path;

// Utility structure to facilitate setting up filesystem structure under test directories.
pub enum TestPathKind {
    File,
    FileWithContents(String),
    Directory(Vec<TestPath>),
}
pub struct TestPath {
    pub name: String,
    pub kind: TestPathKind,
}

impl TestPath {
    pub fn file(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            kind: TestPathKind::File,
        }
    }

    pub fn dir(name: &str, children: Vec<TestPath>) -> Self {
        Self {
            name: name.to_owned(),
            kind: TestPathKind::Directory(children),
        }
    }
}

pub fn setup_test_directory(root: &Path, paths: Vec<TestPath>) {
    for path in paths {
        match path.kind {
            TestPathKind::File => {
                std::fs::File::create(root.join(path.name)).unwrap();
            }
            TestPathKind::Directory(children) => {
                let dir = root.join(path.name);
                std::fs::create_dir(&dir).unwrap();
                setup_test_directory(&dir, children);
            }
            TestPathKind::FileWithContents(contents) => {
                let path = root.join(path.name);
                let mut f = std::fs::File::create(path).unwrap();
                f.write_all(contents.as_bytes()).unwrap();
            }
        }
    }
}
