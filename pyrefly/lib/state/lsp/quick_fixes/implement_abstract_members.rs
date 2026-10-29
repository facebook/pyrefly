/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::DefaultValue;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::literal::Lit;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::type_output::AnnotationPart;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;

/// Builds source text while retaining the module references used by annotations.
struct MemberRenderer<'a> {
    stdlib: &'a Stdlib,
    parts: Vec<AnnotationPart>,
}

impl MemberRenderer<'_> {
    fn text(&mut self, text: &str) {
        if let Some(AnnotationPart::Text(previous)) = self.parts.last_mut() {
            previous.push_str(text);
        } else {
            self.parts.push(AnnotationPart::Text(text.to_owned()));
        }
    }

    fn reference(&mut self, module: &str, name: &str) {
        self.parts.push(AnnotationPart::Reference {
            module: ModuleName::from_str(module),
            name: name.to_owned(),
        });
    }

    fn annotation(&mut self, ty: &Type) {
        self.parts
            .extend(ty.get_annotation_parts(Some(self.stdlib)));
    }

    fn signature(
        &mut self,
        callable: &Callable,
        tparams: Option<&TParams>,
        strip_receiver: bool,
    ) -> Option<()> {
        if let Some(tparams) = tparams.filter(|p| !p.is_empty()) {
            self.text("[");
            for (i, param) in tparams.iter().enumerate() {
                if i > 0 {
                    self.text(", ");
                }
                self.text(&param.display_name_with_prefix().to_string());
                match param.restriction() {
                    Restriction::Bound(ty) => {
                        self.text(": ");
                        self.annotation(ty);
                    }
                    Restriction::Constraints(types) => {
                        self.text(": (");
                        for (i, ty) in types.iter().enumerate() {
                            if i > 0 {
                                self.text(", ");
                            }
                            self.annotation(ty);
                        }
                        self.text(")");
                    }
                    Restriction::Unrestricted => {}
                    Restriction::ShapeExtension(_) => return None,
                }
                if let Some(default) = param.default() {
                    self.text(" = ");
                    self.annotation(default);
                }
            }
            self.text("]");
        }
        self.text("(");
        let Params::List(params) = &callable.params else {
            return None;
        };
        self.params(params, strip_receiver);
        self.text(") -> ");
        self.annotation(&callable.ret);
        Some(())
    }

    fn params(&mut self, params: &ParamList, strip_receiver: bool) {
        let mut posonly = false;
        let mut kwonly = false;
        for (i, param) in params.items().iter().enumerate() {
            if i > 0 {
                self.text(", ");
            }
            if matches!(param, Param::PosOnly(..)) {
                posonly = true;
            } else if posonly {
                self.text("/, ");
                posonly = false;
            }
            if matches!(param, Param::Varargs(..)) {
                kwonly = true;
                self.text("*");
            } else if matches!(param, Param::Kwargs(..)) {
                self.text("**");
            } else if matches!(param, Param::KwOnly(..)) && !kwonly {
                kwonly = true;
                self.text("*, ");
            }
            if let Some(name) = param.name() {
                self.text(name.as_str());
            } else {
                self.text(&format!("arg{i}"));
            }
            if !(strip_receiver && i == 0 && matches!(param, Param::PosOnly(..) | Param::Pos(..))) {
                self.text(": ");
                self.annotation(param.as_type());
            }
            if let Param::PosOnly(_, _, Required::Optional(default))
            | Param::Pos(_, _, Required::Optional(default))
            | Param::KwOnly(_, _, Required::Optional(default)) = param
            {
                self.text(" = ");
                match default.as_ref() {
                    Some(DefaultValue {
                        display: Some(text),
                        ..
                    }) => self.text(text),
                    Some(DefaultValue { ty: Type::None, .. }) => self.text("None"),
                    Some(DefaultValue {
                        ty: Type::Literal(literal),
                        ..
                    }) if !matches!(literal.value, Lit::Enum(_)) => {
                        self.text(&literal.value.to_string());
                    }
                    _ => {
                        // Arbitrary default expressions can depend on names in the defining module.
                        self.reference("typing", "cast");
                        self.text("(");
                        self.reference("typing", "Any");
                        self.text(", ...)");
                    }
                }
            }
        }
        if posonly {
            self.text(", /");
        }
    }
}
