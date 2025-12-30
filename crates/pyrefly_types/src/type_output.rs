/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::qname::QName;

use crate::display::TypeDisplayContext;
use crate::literal::Lit;
use crate::types::TArgs;
use crate::types::Type;

/// A trait that will be will be used for formatting types, but also allow
/// additional functionality. The major difference between implementations
/// of this trait will be the location that type information is written to
/// For example, this will be used to write type information to a formatter as we do now,
/// and also allow us to collect the same types along with their location into a vector.
pub trait TypeOutput {
    fn write_str(&mut self, s: &str) -> fmt::Result;
    fn write_qname(&mut self, qname: &QName) -> fmt::Result;
    fn write_lit(&mut self, lit: &Lit) -> fmt::Result;
    fn write_targs(&mut self, targs: &TArgs) -> fmt::Result;
    fn write_type(&mut self, ty: &Type) -> fmt::Result;
}

/// Implementation of `TypeOutput` that writes formatted types to plain text.
///
/// This struct wraps a `fmt::Formatter` and delegates type formatting to the
/// [`TypeDisplayContext`] to produce typed formatted as plain text.
pub struct DisplayOutput<'a, 'b, 'f> {
    context: &'a TypeDisplayContext<'a>,
    formatter: &'b mut fmt::Formatter<'f>,
}

impl<'a, 'b, 'f> DisplayOutput<'a, 'b, 'f> {
    pub fn new(context: &'a TypeDisplayContext<'a>, formatter: &'b mut fmt::Formatter<'f>) -> Self {
        Self { context, formatter }
    }
}

impl<'a, 'b, 'f> TypeOutput for DisplayOutput<'a, 'b, 'f> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.formatter.write_str(s)
    }

    fn write_qname(&mut self, q: &QName) -> fmt::Result {
        self.context.fmt_qname(q, self.formatter)
    }

    fn write_lit(&mut self, lit: &Lit) -> fmt::Result {
        self.context.fmt_lit(lit, self.formatter)
    }

    fn write_targs(&mut self, targs: &TArgs) -> fmt::Result {
        self.context.fmt_targs(targs, self.formatter)
    }

    fn write_type(&mut self, ty: &Type) -> fmt::Result {
        write!(self.formatter, "{}", self.context.display(ty))
    }
}

/// This struct is used to collect the type to be displayed as a vector. Each element
/// in the vector will be tuple of (String, Option<TextRangeWithModule>).
/// The String the actual part of the string that will be displayed. When displaying the type
/// each of these will be concatenated to create the final type.
/// The second element of the vector is an optional location. For any part that do have
/// a location this will be included. For separators like '|', '[', etc. this will be None.
pub struct OutputWithLocations<'a> {
    parts: Vec<(String, Option<TextRangeWithModule>)>,
    context: &'a TypeDisplayContext<'a>,
}

impl<'a> OutputWithLocations<'a> {
    pub fn new(context: &'a TypeDisplayContext<'a>) -> Self {
        Self {
            parts: Vec::new(),
            context,
        }
    }

    pub fn parts(&self) -> &[(String, Option<TextRangeWithModule>)] {
        &self.parts
    }
}

impl TypeOutput for OutputWithLocations<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.parts.push((s.to_owned(), None));
        Ok(())
    }

    fn write_qname(&mut self, qname: &QName) -> fmt::Result {
        let location = TextRangeWithModule::new(qname.module().clone(), qname.range());
        self.parts.push((qname.id().to_string(), Some(location)));
        Ok(())
    }

    fn write_lit(&mut self, lit: &Lit) -> fmt::Result {
        // Format the literal and extract location if it's an Enum literal
        let formatted = lit.to_string();
        let location = match lit {
            Lit::Enum(lit_enum) => {
                // Enum literals have a class with a qname that has location info
                let qname = lit_enum.class.qname();
                Some(TextRangeWithModule::new(
                    qname.module().clone(),
                    qname.range(),
                ))
            }
            _ => None,
        };
        self.parts.push((formatted, location));
        Ok(())
    }

    fn write_targs(&mut self, targs: &TArgs) -> fmt::Result {
        // Write each type argument separately with its own location
        // This ensures that each type in a union (e.g., int | str) gets its own
        // clickable part with a link to its definition
        if !targs.is_empty() {
            self.write_str("[")?;
            for (i, ty) in targs.as_slice().iter().enumerate() {
                if i > 0 {
                    self.write_str(", ")?;
                }
                self.write_type(ty)?;
            }
            self.write_str("]")?;
        }
        Ok(())
    }

    fn write_type(&mut self, ty: &Type) -> fmt::Result {
        // Format the type and extract location if it has a qname
        self.context.fmt_helper_generic(ty, false, self)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use pyrefly_python::qname::QName;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::lit_int::LitInt;
    use crate::literal::LitEnum;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedKind;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::TArgs;
    use crate::types::TParam;
    use crate::types::TParams;

    fn fake_class(name: &str, module: &str, range: u32) -> Class {
        let mi = Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("1234567890".to_owned()),
        );

        Class::new(
            ClassDefIndex(0),
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(range))),
            NestingContext::toplevel(),
            mi,
            None,
            starlark_map::small_map::SmallMap::new(),
        )
    }

    #[test]
    fn test_output_with_locations_write_str() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        assert_eq!(output.parts().len(), 0);

        output.write_str("hello").unwrap();
        assert_eq!(output.parts().len(), 1);
        assert_eq!(output.parts()[0].0, "hello");
        assert!(output.parts()[0].1.is_none());

        output.write_str(" world").unwrap();
        assert_eq!(output.parts().len(), 2);
        assert_eq!(output.parts()[1].0, " world");
        assert!(output.parts()[1].1.is_none());

        let parts = output.parts();
        assert_eq!(parts[0].0, "hello");
        assert_eq!(parts[1].0, " world");
    }

    #[test]
    fn test_output_with_locations_write_qname() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        let module = Module::new(
            ModuleName::from_str("test_module"),
            ModulePath::filesystem(PathBuf::from("test_module.py")),
            Arc::new("def foo(): pass".to_owned()),
        );

        let identifier = Identifier::new(
            Name::new("MyClass"),
            TextRange::new(TextSize::new(4), TextSize::new(11)),
        );

        let qname = QName::new(identifier, NestingContext::toplevel(), module.clone());
        output.write_qname(&qname).unwrap();

        assert_eq!(output.parts().len(), 1);
        let (name_str, location) = &output.parts()[0];
        assert_eq!(name_str, "MyClass");

        assert!(location.is_some());
        let loc = location.as_ref().unwrap();
        assert_eq!(
            loc.range,
            TextRange::new(TextSize::new(4), TextSize::new(11))
        );
        assert_eq!(loc.module.name(), ModuleName::from_str("test_module"));
    }

    #[test]
    fn test_output_with_locations_write_lit_non_enum() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        // Test with a string literal - should have no location
        let str_lit = Lit::Str("hello".into());
        output.write_lit(&str_lit).unwrap();

        assert_eq!(output.parts().len(), 1);
        assert_eq!(output.parts()[0].0, "'hello'");
        assert!(output.parts()[0].1.is_none());
    }

    #[test]
    fn test_output_with_locations_write_lit_enum() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        // Create an Enum literal with location information
        let enum_class = ClassType::new(fake_class("Color", "colors", 10), TArgs::default());

        let enum_lit = Lit::Enum(Box::new(LitEnum {
            class: enum_class,
            member: Name::new("RED"),
            ty: Type::any_implicit(),
        }));

        output.write_lit(&enum_lit).unwrap();

        assert_eq!(output.parts().len(), 1);
        let (formatted, location) = &output.parts()[0];

        assert_eq!(formatted, "Color.RED");

        // Verify the location was captured from the enum's class qname
        assert!(location.is_some());
        let loc = location.as_ref().unwrap();
        assert_eq!(loc.range, TextRange::empty(TextSize::new(10)));
        assert_eq!(loc.module.name(), ModuleName::from_str("colors"));
    }

    #[test]
    fn test_output_with_locations_write_targs_multiple() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        // Create TArgs with multiple type arguments
        let tparam1 = TParam {
            quantified: Quantified::new(
                pyrefly_util::uniques::UniqueFactory::new().fresh(),
                Name::new("T"),
                QuantifiedKind::TypeVar,
                None,
                Restriction::Unrestricted,
            ),
            variance: PreInferenceVariance::Invariant,
        };
        let tparam2 = TParam {
            quantified: Quantified::new(
                pyrefly_util::uniques::UniqueFactory::new().fresh(),
                Name::new("U"),
                QuantifiedKind::TypeVar,
                None,
                Restriction::Unrestricted,
            ),
            variance: PreInferenceVariance::Invariant,
        };
        let tparam3 = TParam {
            quantified: Quantified::new(
                pyrefly_util::uniques::UniqueFactory::new().fresh(),
                Name::new("V"),
                QuantifiedKind::TypeVar,
                None,
                Restriction::Unrestricted,
            ),
            variance: PreInferenceVariance::Invariant,
        };

        let tparams = Arc::new(TParams::new(vec![tparam1, tparam2, tparam3]));
        let targs = TArgs::new(
            tparams,
            vec![Type::None, Type::LiteralString, Type::any_explicit()],
        );

        output.write_targs(&targs).unwrap();

        // Now that write_type is implemented, it actually writes the types
        // Should have: "[", "None", ", ", "LiteralString", ", ", "Any", "]"
        assert_eq!(output.parts().len(), 7);
        assert_eq!(output.parts()[0].0, "[");
        assert!(output.parts()[0].1.is_none());

        assert_eq!(output.parts()[1].0, "None");
        assert!(output.parts()[1].1.is_none());

        assert_eq!(output.parts()[2].0, ", ");
        assert!(output.parts()[2].1.is_none());

        assert_eq!(output.parts()[3].0, "LiteralString");
        assert!(output.parts()[3].1.is_none());

        assert_eq!(output.parts()[4].0, ", ");
        assert!(output.parts()[4].1.is_none());

        assert_eq!(output.parts()[5].0, "Any");
        assert!(output.parts()[5].1.is_none());

        assert_eq!(output.parts()[6].0, "]");
        assert!(output.parts()[6].1.is_none());
    }

    #[test]
    fn test_output_with_locations_write_type_simple() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        // Test simple types that don't have locations
        output.write_type(&Type::None).unwrap();
        assert_eq!(output.parts().len(), 1);
        assert_eq!(output.parts()[0].0, "None");
        assert!(output.parts()[0].1.is_none());

        output.write_type(&Type::LiteralString).unwrap();
        assert_eq!(output.parts().len(), 2);
        assert_eq!(output.parts()[1].0, "LiteralString");
        assert!(output.parts()[1].1.is_none());
    }

    #[test]
    fn test_output_with_locations_write_type_tuple() {
        // Test tuple[int, str]
        let int_class = fake_class("int", "builtins", 30);
        let str_class = fake_class("str", "builtins", 40);

        let int_type = Type::ClassType(ClassType::new(int_class, TArgs::default()));
        let str_type = Type::ClassType(ClassType::new(str_class, TArgs::default()));
        let tuple_type = Type::Tuple(Tuple::Concrete(vec![int_type.clone(), str_type.clone()]));

        let context = TypeDisplayContext::new(&[&tuple_type, &int_type, &str_type]);
        let mut output = OutputWithLocations::new(&context);

        output.write_type(&tuple_type).unwrap();
        assert!(!output.parts().is_empty());

        // Find the int and str parts and verify they have locations
        let int_part = output.parts().iter().find(|p| p.0 == "int");
        assert!(int_part.is_some());
        assert!(int_part.unwrap().1.is_some());

        let str_part = output.parts().iter().find(|p| p.0 == "str");
        assert!(str_part.is_some());
        assert!(str_part.unwrap().1.is_some());
    }

    #[test]
    fn test_output_with_locations_union_type_splits_properly() {
        // Create int | str | None type
        let int_class = fake_class("int", "builtins", 10);
        let str_class = fake_class("str", "builtins", 20);

        let int_type = Type::ClassType(ClassType::new(int_class, TArgs::default()));
        let str_type = Type::ClassType(ClassType::new(str_class, TArgs::default()));
        let union_type = Type::union(vec![int_type, str_type, Type::None]);

        let ctx = TypeDisplayContext::new(&[&union_type]);
        let mut output = OutputWithLocations::new(&ctx);

        ctx.fmt_helper_generic(&union_type, false, &mut output)
            .unwrap();

        let parts_str: String = output.parts().iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(parts_str, "int | str | None");

        // New behavior: Union types are split into separate parts
        // Expected: [("int", Some(location)), (" | ", None), ("str", Some(location)), (" | ", None), ("None", None)]
        let parts = output.parts();
        assert_eq!(parts.len(), 5, "Union should be split into 5 parts");

        // Verify each part
        assert_eq!(parts[0].0, "int");
        assert!(parts[0].1.is_some(), "int should have location");

        assert_eq!(parts[1].0, " | ");
        assert!(parts[1].1.is_none(), "separator should not have location");

        assert_eq!(parts[2].0, "str");
        assert!(parts[2].1.is_some(), "str should have location");

        assert_eq!(parts[3].0, " | ");
        assert!(parts[3].1.is_none(), "separator should not have location");

        assert_eq!(parts[4].0, "None");
        assert!(parts[4].1.is_none(), "None should not have location");
    }

    #[test]
    fn test_output_with_locations_intersection_type_splits_properly() {
        // Create int & str type (doesn't make sense semantically, but tests the formatting)
        let int_type = Type::ClassType(ClassType::new(
            fake_class("int", "builtins", 10),
            TArgs::default(),
        ));
        let str_type = Type::ClassType(ClassType::new(
            fake_class("str", "builtins", 20),
            TArgs::default(),
        ));
        let intersect_type =
            Type::Intersect(Box::new((vec![int_type, str_type], Type::any_implicit())));

        let ctx = TypeDisplayContext::new(&[&intersect_type]);
        let mut output = OutputWithLocations::new(&ctx);

        // Format the type using fmt_helper_generic
        ctx.fmt_helper_generic(&intersect_type, false, &mut output)
            .unwrap();

        // Check the concatenated result
        let parts_str: String = output.parts().iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(parts_str, "int & str");

        // New behavior: Intersection types are split into separate parts
        // Expected: [("int", Some(location)), (" & ", None), ("str", Some(location))]
        let parts = output.parts();
        assert_eq!(parts.len(), 3, "Intersection should be split into 3 parts");

        // Verify each part
        assert_eq!(parts[0].0, "int");
        assert!(parts[0].1.is_some(), "int should have location");

        assert_eq!(parts[1].0, " & ");
        assert!(parts[1].1.is_none(), "separator should not have location");

        assert_eq!(parts[2].0, "str");
        assert!(parts[2].1.is_some(), "str should have location");
    }

    #[test]
    fn test_output_with_locations_tuple_base_not_clickable() {
        // TODO(jvansch): When implementing clickable support for the base type in generics like tuple[int],
        // update this test to verify that "tuple" has a location and is clickable.
        // Expected future behavior: [("tuple", Some(location)), ("[", None), ("int", Some(location)), ("]", None)]

        // Create tuple[int] type
        let int_class = fake_class("int", "builtins", 10);
        let int_type = Type::ClassType(ClassType::new(int_class, TArgs::default()));
        let tuple_type = Type::Tuple(Tuple::Concrete(vec![int_type]));

        let ctx = TypeDisplayContext::new(&[&tuple_type]);
        let mut output = OutputWithLocations::new(&ctx);

        ctx.fmt_helper_generic(&tuple_type, false, &mut output)
            .unwrap();

        let parts_str: String = output.parts().iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(parts_str, "tuple[int]");

        // Current behavior: The "tuple" part is NOT clickable
        // Expected parts: [("tuple", None), ("[", None), ("int", Some(location)), ("]", None)]
        let parts = output.parts();
        assert_eq!(parts.len(), 4, "Should have 4 parts");

        // Verify each part
        assert_eq!(parts[0].0, "tuple");
        assert!(
            parts[0].1.is_none(),
            "tuple[ should not have location (not clickable)"
        );

        assert_eq!(parts[1].0, "[");
        assert!(parts[1].1.is_none(), "[ should not have location");

        assert_eq!(parts[2].0, "int");
        assert!(parts[2].1.is_some(), "int should have location (clickable)");

        assert_eq!(parts[3].0, "]");
        assert!(parts[3].1.is_none(), "] should not have location");
    }

    #[test]
    fn test_output_with_locations_literal_base_not_clickable() {
        // TODO(jvansch): When implementing clickable support for the base type in special forms like Literal[1],
        // update this test to verify that "Literal" has a location and is clickable.
        // Expected future behavior: [("Literal", Some(location)), ("[", None), ("1", None), ("]", None)]

        // Create Literal[1] type
        let literal_type = Type::Literal(Lit::Int(LitInt::new(1)));

        let ctx = TypeDisplayContext::new(&[&literal_type]);
        let mut output = OutputWithLocations::new(&ctx);

        ctx.fmt_helper_generic(&literal_type, false, &mut output)
            .unwrap();

        let parts_str: String = output.parts().iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(parts_str, "Literal[1]");

        // Current behavior: The "Literal" part is NOT clickable
        // Expected parts: [("Literal", None), ("[", None), ("1", None), ("]", None)]
        let parts = output.parts();
        assert_eq!(parts.len(), 4, "Should have 4 parts");

        // Verify each part
        assert_eq!(parts[0].0, "Literal");
        assert!(
            parts[0].1.is_none(),
            "Literal should not have location (not clickable)"
        );

        assert_eq!(parts[1].0, "[");
        assert!(parts[1].1.is_none(), "[ should not have location");

        assert_eq!(parts[2].0, "1");
        assert!(parts[2].1.is_none(), "1 should not have location");

        assert_eq!(parts[3].0, "]");
        assert!(parts[3].1.is_none(), "] should not have location");
    }
}
