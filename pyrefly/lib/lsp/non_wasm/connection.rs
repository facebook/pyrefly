/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::BufReader;
use std::io::Stdin;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;

use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::read_lsp_message;
use crate::lsp::non_wasm::protocol::write_lsp_message;

pub struct Connection {
    pub sender: Sender<Message>,
    /// Channel receiver, only present for test connections created via
    /// `Connection::memory()`. The test client reads from this to observe
    /// messages sent by the server.
    channel_receiver: Option<Receiver<Message>>,
}

/// Owns the message source for the LSP/TSP server. Either a crossbeam channel
/// (used in tests via `Connection::memory()`) or a direct stdin reader (used in
/// production via `Connection::stdio()`).
///
/// This is kept separate from `Connection` so the read side can take `&mut self`
/// without requiring interior mutability — stdin is only ever read from one
/// thread.
pub enum MessageReader {
    Channel(Receiver<Message>),
    Stdio(BufReader<Stdin>),
    /// A generic byte stream, used for IPC transports (Unix domain sockets,
    /// Windows named pipes).
    Stream(BufReader<Box<dyn std::io::Read + Send>>),
}

impl MessageReader {
    /// Receive the next message, blocking until one is available.
    /// Returns `None` if the connection is closed (channel disconnected or
    /// stdin EOF).
    pub fn recv(&mut self) -> Option<Message> {
        match self {
            MessageReader::Channel(r) => r.recv().ok(),
            MessageReader::Stdio(r) => read_lsp_message(r).ok().flatten(),
            MessageReader::Stream(r) => read_lsp_message(r).ok().flatten(),
        }
    }
}

pub struct IoThread {
    writer: JoinHandle<std::io::Result<()>>,
}

impl IoThread {
    pub fn join(self) -> std::io::Result<()> {
        match self.writer.join() {
            Ok(result) => result,
            Err(e) => std::panic::panic_any(e),
        }
    }
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WindowsNamedPipeExpectedAccess {
    Read,
    Write,
}

#[cfg(windows)]
impl WindowsNamedPipeExpectedAccess {
    fn open(self, pipe_name: &str) -> std::io::Result<std::fs::File> {
        use std::fs::OpenOptions;

        let mut options = OpenOptions::new();
        match self {
            Self::Read => {
                options.read(true);
            }
            Self::Write => {
                options.write(true);
            }
        }
        options.open(pipe_name)
    }
}

#[cfg(windows)]
fn open_windows_split_pipe_with<T>(
    expected_access: WindowsNamedPipeExpectedAccess,
    open_duplex: impl FnOnce() -> std::io::Result<T>,
    open_expected: impl FnOnce(WindowsNamedPipeExpectedAccess) -> std::io::Result<T>,
) -> std::io::Result<T> {
    open_duplex().or_else(|_| open_expected(expected_access))
}

impl Connection {
    fn from_ipc_streams(
        writer_stream: Box<dyn Write + Send>,
        reader_stream: Box<dyn std::io::Read + Send>,
    ) -> (Self, MessageReader, IoThread) {
        let (writer_sender, writer_receiver) = crossbeam_channel::unbounded();
        let writer = std::thread::spawn(move || {
            let mut output = writer_stream;
            while let Ok(msg) = writer_receiver.recv() {
                write_lsp_message(&mut output, msg)?;
            }
            Ok(())
        });
        (
            Self {
                sender: writer_sender,
                channel_receiver: None,
            },
            MessageReader::Stream(BufReader::new(Box::new(reader_stream))),
            IoThread { writer },
        )
    }

    /// Create a connection that reads directly from stdin and writes to stdout.
    /// Only the writer uses a background thread; reads happen inline in the
    /// calling thread, eliminating a context switch per LSP message.
    pub fn stdio() -> (Self, MessageReader, IoThread) {
        let (writer_sender, writer_receiver) = crossbeam_channel::unbounded();
        let writer = std::thread::spawn(move || {
            let mut stdout = std::io::stdout().lock();
            while let Ok(msg) = writer_receiver.recv() {
                write_lsp_message(&mut stdout, msg)?
            }
            Ok(())
        });
        (
            Self {
                sender: writer_sender,
                channel_receiver: None,
            },
            MessageReader::Stdio(BufReader::new(std::io::stdin())),
            IoThread { writer },
        )
    }

    /// Create a connection over a local IPC mechanism (Unix domain socket on
    /// Unix, named pipe on Windows). The `pipe_name` is a socket path on Unix
    /// or a pipe name on Windows (automatically prefixed with `\\.\pipe\`).
    pub fn ipc(pipe_name: &str) -> std::io::Result<(Self, MessageReader, IoThread)> {
        let (writer_stream, reader_stream) = Self::connect_ipc(pipe_name)?;
        Ok(Self::from_ipc_streams(writer_stream, reader_stream))
    }

    /// Create a connection over IPC endpoints that may be provided either as
    /// one full-duplex endpoint or as separate inbound and outbound endpoints.
    pub fn ipc_split(
        input_pipe_name: &str,
        output_pipe_name: &str,
    ) -> std::io::Result<(Self, MessageReader, IoThread)> {
        let writer_stream = Self::connect_ipc_writer(output_pipe_name)?;
        let reader_stream = Self::connect_ipc_reader(input_pipe_name)?;
        Ok(Self::from_ipc_streams(writer_stream, reader_stream))
    }

    #[cfg(unix)]
    fn connect_ipc(
        pipe_name: &str,
    ) -> std::io::Result<(Box<dyn Write + Send>, Box<dyn std::io::Read + Send>)> {
        let stream = UnixStream::connect(pipe_name)?;
        let reader = stream.try_clone()?;
        Ok((Box::new(stream), Box::new(reader)))
    }

    #[cfg(unix)]
    fn connect_ipc_reader(pipe_name: &str) -> std::io::Result<Box<dyn std::io::Read + Send>> {
        Ok(Box::new(UnixStream::connect(pipe_name)?))
    }

    #[cfg(unix)]
    fn connect_ipc_writer(pipe_name: &str) -> std::io::Result<Box<dyn Write + Send>> {
        Ok(Box::new(UnixStream::connect(pipe_name)?))
    }

    #[cfg(windows)]
    fn connect_ipc(
        pipe_name: &str,
    ) -> std::io::Result<(Box<dyn Write + Send>, Box<dyn std::io::Read + Send>)> {
        let stream = Self::open_windows_named_pipe(pipe_name)?;
        let reader = stream.try_clone()?;
        Ok((Box::new(stream), Box::new(reader)))
    }

    #[cfg(windows)]
    fn connect_ipc_reader(pipe_name: &str) -> std::io::Result<Box<dyn std::io::Read + Send>> {
        Ok(Box::new(Self::open_windows_split_named_pipe(
            pipe_name,
            WindowsNamedPipeExpectedAccess::Read,
        )?))
    }

    #[cfg(windows)]
    fn connect_ipc_writer(pipe_name: &str) -> std::io::Result<Box<dyn Write + Send>> {
        Ok(Box::new(Self::open_windows_split_named_pipe(
            pipe_name,
            WindowsNamedPipeExpectedAccess::Write,
        )?))
    }

    #[cfg(windows)]
    fn open_windows_named_pipe(pipe_name: &str) -> std::io::Result<std::fs::File> {
        use std::fs::OpenOptions;

        OpenOptions::new().read(true).write(true).open(pipe_name)
    }

    #[cfg(windows)]
    fn open_windows_split_named_pipe(
        pipe_name: &str,
        expected_access: WindowsNamedPipeExpectedAccess,
    ) -> std::io::Result<std::fs::File> {
        // This workaround is needed because of a few limitations on Windows.
        //
        // Windows named pipes do not support synchronous concurrent reads and
        // writes on the same handle, so we cannot use a single named pipe for
        // full-duplex communication here. Windows does support concurrent
        // reads and writes through async APIs, but Pyrefly has a synchronous
        // codebase, so that approach is not available to us.
        //
        // There is also a Node.js limitation. To work around the Windows
        // limitation, we support using two IPC channels separately for inbound
        // and outbound communication, similar to stdin and stdout in stdio.
        // However, Node.js does not allow creating a Windows IPC named pipe
        // that is only inbound or only outbound; it always creates a
        // full-duplex pipe. Also, we cannot know how the pipe was created
        // until we open it.
        //
        // Because of these two constraints, we use this workaround on
        // Windows: we first try to open the pipe with both read and write
        // access. If that fails, we open it only with the expected access for
        // clients that can create a named pipe with specific access permissions.
        open_windows_split_pipe_with(
            expected_access,
            || Self::open_windows_named_pipe(pipe_name),
            |expected_access| expected_access.open(pipe_name),
        )
    }

    /// Create a connection from a transport specification string.
    /// Supported values: `"stdio"` for stdin/stdout, or `"ipc://<name>"` for a
    /// local socket / named pipe.
    pub fn from_transport(transport: &str) -> std::io::Result<(Self, MessageReader, IoThread)> {
        if transport == "stdio" {
            return Ok(Self::stdio());
        }

        if let Some(pipe_name) = transport.strip_prefix("ipc://") {
            return Self::ipc(pipe_name);
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Unsupported TSP transport: {transport}"),
        ))
    }

    pub fn memory() -> ((Self, MessageReader), (Self, MessageReader)) {
        let (s1, r1) = crossbeam_channel::unbounded();
        let (s2, r2) = crossbeam_channel::unbounded();
        (
            (
                Self {
                    sender: s1,
                    channel_receiver: Some(r2.clone()),
                },
                MessageReader::Channel(r2),
            ),
            (
                Self {
                    sender: s2,
                    channel_receiver: Some(r1.clone()),
                },
                MessageReader::Channel(r1),
            ),
        )
    }

    /// Access the underlying channel receiver. Only available for
    /// channel-based connections (tests).
    pub fn channel_receiver(&self) -> &Receiver<Message> {
        self.channel_receiver
            .as_ref()
            .expect("channel_receiver not available for stdio connections")
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use crate::lsp::non_wasm::protocol::Message;
    use crate::lsp::non_wasm::protocol::Notification;

    fn notification_message(method: &str) -> Message {
        Message::Notification(Notification {
            method: method.to_owned(),
            params: serde_json::Value::Null,
            activity_key: None,
        })
    }

    fn assert_notification_method(message: Message, expected: &str) {
        match message {
            Message::Notification(notification) => {
                assert_eq!(notification.method, expected);
            }
            other => panic!("expected notification, got {other:?}"),
        }
    }

    fn expect_io_error<T>(result: io::Result<T>, message: &str) -> io::Error {
        match result {
            Ok(_) => panic!("{message}"),
            Err(error) => error,
        }
    }

    #[cfg(any(unix, windows))]
    mod ipc_transport {
        use std::io;
        use std::io::BufReader;
        use std::thread;

        use super::super::Connection;
        use super::assert_notification_method;
        use super::expect_io_error;
        use super::notification_message;
        use crate::lsp::non_wasm::protocol::Message;
        use crate::lsp::non_wasm::protocol::read_lsp_message;
        use crate::lsp::non_wasm::protocol::write_lsp_message;

        #[derive(Clone, Copy)]
        enum SplitEndpointMode {
            Duplex,
            Directional,
        }

        #[cfg(unix)]
        mod platform {
            use std::io;
            use std::net::Shutdown;
            use std::os::unix::net::UnixListener;
            use std::os::unix::net::UnixStream;

            use tempfile::TempDir;

            use super::SplitEndpointMode;

            pub type PeerStream = UnixStream;

            pub struct SingleEndpoint {
                pub name: String,
                listener: UnixListener,
                _tempdir: TempDir,
            }

            pub struct SplitEndpoint {
                pub input_name: String,
                pub output_name: String,
                mode: SplitEndpointMode,
                input_listener: UnixListener,
                output_listener: UnixListener,
                _tempdir: TempDir,
            }

            pub struct MissingSplitOutputEndpoint {
                pub input_name: String,
                pub output_name: String,
                _input_listener: UnixListener,
                _tempdir: TempDir,
            }

            fn socket_name(tempdir: &TempDir, name: &str) -> String {
                tempdir
                    .path()
                    .join(name)
                    .to_str()
                    .expect("temporary Unix socket path should be valid UTF-8")
                    .to_owned()
            }

            pub fn single_endpoint(_label: &str) -> io::Result<SingleEndpoint> {
                let tempdir = tempfile::tempdir()?;
                let name = socket_name(&tempdir, "single.sock");
                let listener = UnixListener::bind(&name)?;
                Ok(SingleEndpoint {
                    name,
                    listener,
                    _tempdir: tempdir,
                })
            }

            pub fn accept_single_endpoint(endpoint: SingleEndpoint) -> io::Result<PeerStream> {
                let (stream, _) = endpoint.listener.accept()?;
                Ok(stream)
            }

            pub fn split_endpoint(
                label: &str,
                mode: SplitEndpointMode,
            ) -> io::Result<SplitEndpoint> {
                let tempdir = tempfile::tempdir()?;
                let input_name = socket_name(&tempdir, &format!("{label}-i.sock"));
                let output_name = socket_name(&tempdir, &format!("{label}-o.sock"));
                let input_listener = UnixListener::bind(&input_name)?;
                let output_listener = UnixListener::bind(&output_name)?;
                Ok(SplitEndpoint {
                    input_name,
                    output_name,
                    mode,
                    input_listener,
                    output_listener,
                    _tempdir: tempdir,
                })
            }

            pub fn accept_split_endpoint(
                endpoint: SplitEndpoint,
            ) -> io::Result<(PeerStream, PeerStream)> {
                let (output_stream, _) = endpoint.output_listener.accept()?;
                let (input_stream, _) = endpoint.input_listener.accept()?;
                match endpoint.mode {
                    SplitEndpointMode::Duplex => {}
                    SplitEndpointMode::Directional => {
                        input_stream.shutdown(Shutdown::Read)?;
                        output_stream.shutdown(Shutdown::Write)?;
                    }
                }
                Ok((input_stream, output_stream))
            }

            pub fn missing_endpoint_name(_label: &str) -> io::Result<String> {
                let tempdir = tempfile::tempdir()?;
                Ok(socket_name(&tempdir, "missing.sock"))
            }

            pub fn missing_split_output_endpoint() -> io::Result<MissingSplitOutputEndpoint> {
                let tempdir = tempfile::tempdir()?;
                let input_name = socket_name(&tempdir, "input.sock");
                let output_name = socket_name(&tempdir, "missing-output.sock");
                let input_listener = UnixListener::bind(&input_name)?;
                Ok(MissingSplitOutputEndpoint {
                    input_name,
                    output_name,
                    _input_listener: input_listener,
                    _tempdir: tempdir,
                })
            }
        }

        #[cfg(windows)]
        mod platform {
            use std::ffi::OsStr;
            use std::ffi::c_void;
            use std::fs::File;
            use std::io;
            use std::os::windows::ffi::OsStrExt;
            use std::os::windows::io::AsRawHandle;
            use std::os::windows::io::FromRawHandle;
            use std::ptr::null_mut;

            use uuid::Uuid;

            use super::SplitEndpointMode;

            pub type PeerStream = File;

            pub struct SingleEndpoint {
                pub name: String,
                pipe: File,
            }

            pub struct SplitEndpoint {
                pub input_name: String,
                pub output_name: String,
                input_pipe: File,
                output_pipe: File,
            }

            pub struct MissingSplitOutputEndpoint {
                pub input_name: String,
                pub output_name: String,
                _input_pipe: File,
            }

            const ERROR_PIPE_CONNECTED: i32 = 535;
            const PIPE_ACCESS_INBOUND: u32 = 0x0000_0001;
            const PIPE_ACCESS_OUTBOUND: u32 = 0x0000_0002;
            const PIPE_ACCESS_DUPLEX: u32 = 0x0000_0003;
            const PIPE_TYPE_BYTE: u32 = 0x0000_0000;
            const PIPE_READMODE_BYTE: u32 = 0x0000_0000;
            const PIPE_WAIT: u32 = 0x0000_0000;
            const INVALID_HANDLE_VALUE: *mut c_void = -1_isize as *mut c_void;

            unsafe extern "system" {
                fn CreateNamedPipeW(
                    lp_name: *const u16,
                    dw_open_mode: u32,
                    dw_pipe_mode: u32,
                    n_max_instances: u32,
                    n_out_buffer_size: u32,
                    n_in_buffer_size: u32,
                    n_default_time_out: u32,
                    lp_security_attributes: *mut c_void,
                ) -> *mut c_void;

                fn ConnectNamedPipe(h_named_pipe: *mut c_void, lp_overlapped: *mut c_void) -> i32;
            }

            fn pipe_name(label: &str) -> String {
                format!(
                    r"\\.\pipe\pyrefly-{label}-{}-{}",
                    std::process::id(),
                    Uuid::new_v4()
                )
            }

            fn wide_null(value: &str) -> Vec<u16> {
                OsStr::new(value).encode_wide().chain(Some(0)).collect()
            }

            fn create_named_pipe(pipe_name: &str, access: u32) -> io::Result<File> {
                let pipe_name = wide_null(pipe_name);
                let handle = unsafe {
                    CreateNamedPipeW(
                        pipe_name.as_ptr(),
                        access,
                        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                        1,
                        4096,
                        4096,
                        0,
                        null_mut(),
                    )
                };
                if handle == INVALID_HANDLE_VALUE {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(unsafe { File::from_raw_handle(handle) })
                }
            }

            fn connect_named_pipe(pipe: &File) -> io::Result<()> {
                let connected = unsafe { ConnectNamedPipe(pipe.as_raw_handle(), null_mut()) };
                if connected != 0 {
                    return Ok(());
                }
                let error = io::Error::last_os_error();
                if error.raw_os_error() == Some(ERROR_PIPE_CONNECTED) {
                    Ok(())
                } else {
                    Err(error)
                }
            }

            pub fn single_endpoint(label: &str) -> io::Result<SingleEndpoint> {
                let name = pipe_name(label);
                let pipe = create_named_pipe(&name, PIPE_ACCESS_DUPLEX)?;
                Ok(SingleEndpoint { name, pipe })
            }

            pub fn accept_single_endpoint(endpoint: SingleEndpoint) -> io::Result<PeerStream> {
                connect_named_pipe(&endpoint.pipe)?;
                Ok(endpoint.pipe)
            }

            pub fn split_endpoint(
                label: &str,
                mode: SplitEndpointMode,
            ) -> io::Result<SplitEndpoint> {
                let (input_access, output_access) = match mode {
                    SplitEndpointMode::Duplex => (PIPE_ACCESS_DUPLEX, PIPE_ACCESS_DUPLEX),
                    SplitEndpointMode::Directional => (PIPE_ACCESS_OUTBOUND, PIPE_ACCESS_INBOUND),
                };
                let input_name = pipe_name(&format!("{label}-input"));
                let output_name = pipe_name(&format!("{label}-output"));
                let input_pipe = create_named_pipe(&input_name, input_access)?;
                let output_pipe = create_named_pipe(&output_name, output_access)?;
                Ok(SplitEndpoint {
                    input_name,
                    output_name,
                    input_pipe,
                    output_pipe,
                })
            }

            pub fn accept_split_endpoint(
                endpoint: SplitEndpoint,
            ) -> io::Result<(PeerStream, PeerStream)> {
                connect_named_pipe(&endpoint.output_pipe)?;
                connect_named_pipe(&endpoint.input_pipe)?;
                Ok((endpoint.input_pipe, endpoint.output_pipe))
            }

            pub fn missing_endpoint_name(label: &str) -> io::Result<String> {
                Ok(pipe_name(label))
            }

            pub fn missing_split_output_endpoint() -> io::Result<MissingSplitOutputEndpoint> {
                let input_name = pipe_name("existing-input");
                let output_name = pipe_name("missing-output");
                let input_pipe = create_named_pipe(&input_name, PIPE_ACCESS_OUTBOUND)?;
                Ok(MissingSplitOutputEndpoint {
                    input_name,
                    output_name,
                    _input_pipe: input_pipe,
                })
            }
        }

        fn read_message_from_peer(
            stream: platform::PeerStream,
            missing_message: &str,
        ) -> io::Result<Message> {
            let mut reader = BufReader::new(stream);
            read_lsp_message(&mut reader)?
                .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, missing_message))
        }

        fn run_split_endpoint_test(mode: SplitEndpointMode, label: &str) -> io::Result<()> {
            let endpoint = platform::split_endpoint(label, mode)?;
            let input_name = endpoint.input_name.clone();
            let output_name = endpoint.output_name.clone();
            let peer = thread::spawn(move || -> io::Result<Message> {
                let (mut input_stream, output_stream) = platform::accept_split_endpoint(endpoint)?;
                write_lsp_message(&mut input_stream, notification_message("client/to/server"))?;
                read_message_from_peer(output_stream, "expected a message on the output endpoint")
            });

            let (connection, mut reader, io_thread) =
                Connection::ipc_split(&input_name, &output_name)?;
            assert_notification_method(
                reader
                    .recv()
                    .expect("expected a message from the input endpoint"),
                "client/to/server",
            );

            connection
                .sender
                .send(notification_message("server/to/client"))
                .expect("test connection should stay open");
            drop(connection);
            io_thread.join()?;

            let outbound = peer.join().expect("split IPC peer panicked")?;
            assert_notification_method(outbound, "server/to/client");
            Ok(())
        }

        #[test]
        fn test_ipc_single_endpoint_is_full_duplex() -> io::Result<()> {
            let endpoint = platform::single_endpoint("single")?;
            let name = endpoint.name.clone();
            let peer = thread::spawn(move || -> io::Result<Message> {
                let mut stream = platform::accept_single_endpoint(endpoint)?;
                write_lsp_message(&mut stream, notification_message("client/to/server"))?;
                read_message_from_peer(stream, "expected a message from the server writer")
            });

            let (connection, mut reader, io_thread) = Connection::ipc(&name)?;
            assert_notification_method(
                reader.recv().expect("expected a message from the peer"),
                "client/to/server",
            );

            connection
                .sender
                .send(notification_message("server/to/client"))
                .expect("test connection should stay open");
            drop(connection);
            io_thread.join()?;

            let outbound = peer.join().expect("IPC peer panicked")?;
            assert_notification_method(outbound, "server/to/client");
            Ok(())
        }

        #[test]
        fn test_ipc_split_uses_duplex_endpoints() -> io::Result<()> {
            run_split_endpoint_test(SplitEndpointMode::Duplex, "duplex")
        }

        #[test]
        fn test_ipc_split_uses_directional_endpoints() -> io::Result<()> {
            run_split_endpoint_test(SplitEndpointMode::Directional, "directional")
        }

        #[test]
        fn test_ipc_single_endpoint_reports_missing_endpoint() -> io::Result<()> {
            let name = platform::missing_endpoint_name("missing-single")?;

            let error = expect_io_error(Connection::ipc(&name), "missing endpoint should fail");

            assert_eq!(error.kind(), io::ErrorKind::NotFound);
            Ok(())
        }

        #[test]
        fn test_ipc_split_reports_missing_output_endpoint() -> io::Result<()> {
            let endpoint = platform::missing_split_output_endpoint()?;

            let error = expect_io_error(
                Connection::ipc_split(&endpoint.input_name, &endpoint.output_name),
                "missing output endpoint should fail",
            );

            assert_eq!(error.kind(), io::ErrorKind::NotFound);
            Ok(())
        }
    }
}
