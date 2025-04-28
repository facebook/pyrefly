use ruff_source_file::SourceLocation;
use ruff_text_size::TextSize;

use crate::module::module_info::ModuleInfo;
use crate::module::module_info::SourceRange;

pub fn source_range_to_range(x: &SourceRange) -> lsp_types::Range {
    lsp_types::Range::new(
        source_location_to_position(&x.start),
        source_location_to_position(&x.end),
    )
}

fn source_location_to_position(x: &SourceLocation) -> lsp_types::Position {
    lsp_types::Position {
        line: x.row.to_zero_indexed() as u32,
        character: x.column.to_zero_indexed() as u32,
    }
}

pub fn text_size_to_position(info: &ModuleInfo, x: TextSize) -> lsp_types::Position {
    source_location_to_position(&info.source_location(x))
}

pub fn position_to_text_size(info: &ModuleInfo, position: lsp_types::Position) -> TextSize {
    info.to_text_size(position.line, position.character)
}
