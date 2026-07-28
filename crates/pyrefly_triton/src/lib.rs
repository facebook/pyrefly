pub mod inference;

use ruff_python_ast::StmtFunctionDef;

pub fn check_triton_function(func_def: &StmtFunctionDef) -> Result<(), String> {
    if func_def.name.is_empty() {
        return Err("Function name is empty".to_string());
    }
    
    for stmt in &func_def.body {
        inference::visit_stmt(stmt)?;
    }
    Ok(())
}
