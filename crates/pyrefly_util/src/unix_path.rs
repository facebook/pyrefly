use std::path::Path;

pub fn path_to_unix_string(path: &Path) -> String {
    let mut path_str = path.to_string_lossy().into_owned();
    if std::path::MAIN_SEPARATOR != '/' {
        path_str = path_str.replace(std::path::MAIN_SEPARATOR, "/");
    }
    path_str
}
