use std::path::PathBuf;

use pyo3::prelude::*;
use pyrefly::api::Checker as RustChecker;

#[pyclass(frozen, get_all)]
struct Diagnostic {
    kind: String,
    message: String,
    line: u32,
    column: u32,
}

#[pymethods]
impl Diagnostic {
    fn __repr__(&self) -> String {
        format!(
            "Diagnostic(kind={:?}, message={:?}, line={}, column={})",
            self.kind, self.message, self.line, self.column
        )
    }
}

#[pyclass]
struct Checker {
    inner: RustChecker,
}

#[pymethods]
impl Checker {
    #[new]
    fn new(project_root: PathBuf) -> Self {
        Self {
            inner: RustChecker::new(&project_root),
        }
    }

    fn check(&mut self, code: &str) -> Vec<Diagnostic> {
        self.inner
            .check(code)
            .into_iter()
            .map(|d| Diagnostic {
                kind: d.kind.to_name().to_owned(),
                message: d.message,
                line: d.line,
                column: d.column,
            })
            .collect()
    }
}

#[pymodule]
fn pyrefly_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Checker>()?;
    m.add_class::<Diagnostic>()?;
    Ok(())
}
