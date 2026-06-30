#[cfg(test)]
mod tests {
    use std::fs;

    use crate::api::Checker;
    use crate::config::error_kind::ErrorKind;

    #[test]
    fn test_checker_checks_snippet_against_project_tree() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("helper.py"), "def greet() -> int: ...\n").unwrap();

        let mut checker = Checker::new(root.path());

        // should return err
        let errors = checker.check("from helper import greet\nx: str = greet()");
        assert_eq!(
            errors.iter().map(|e| e.kind).collect::<Vec<_>>(),
            vec![ErrorKind::BadAssignment],
        );

        // now this should work
        assert!(checker.check("from helper import greet\ny: int = greet()").is_empty());
    }
}