// @generated
// @codegen-command: cd ~/fbsource/fbcode/pyrefly && ./facebook/generate_pysa_report_capnp.sh
// @codegen-source: fbcode/pyrefly/pyrefly/lib/report/pysa/pysa_report.capnp
#![cfg_attr(rustfmt, rustfmt_skip)]

// DO NOT EDIT.
// source: pyrefly/lib/report/pysa/pysa_report.capnp


pub mod source_path {
  pub use self::Which::{FileSystem,Namespace,Memory,BundledTypeshed,BundledTypeshedThirdParty,BundledThirdParty};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_file_system(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_namespace(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 1 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_memory(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 2 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_bundled_typeshed(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 3 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_bundled_typeshed_third_party(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 4 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_bundled_third_party(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 5 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(FileSystem(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Namespace(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(Memory(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        3 => {
          ::core::result::Result::Ok(BundledTypeshed(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        4 => {
          ::core::result::Result::Ok(BundledTypeshedThirdParty(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        5 => {
          ::core::result::Result::Ok(BundledThirdParty(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_file_system(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_file_system(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 0);
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_file_system(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_namespace(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_namespace(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 1);
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_namespace(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 1 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_memory(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      self.builder.set_data_field::<u16>(0, 2);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_memory(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 2);
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_memory(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 2 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_bundled_typeshed(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      self.builder.set_data_field::<u16>(0, 3);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_bundled_typeshed(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 3);
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_bundled_typeshed(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 3 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_bundled_typeshed_third_party(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      self.builder.set_data_field::<u16>(0, 4);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_bundled_typeshed_third_party(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 4);
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_bundled_typeshed_third_party(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 4 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_bundled_third_party(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      self.builder.set_data_field::<u16>(0, 5);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_bundled_third_party(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 5);
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_bundled_third_party(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 5 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(FileSystem(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Namespace(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(Memory(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        3 => {
          ::core::result::Result::Ok(BundledTypeshed(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        4 => {
          ::core::result::Result::Ok(BundledTypeshedThirdParty(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        5 => {
          ::core::result::Result::Ok(BundledThirdParty(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 119] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(125, 96, 95, 20, 53, 150, 120, 201),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 6, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 170, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 87, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 83, 111, 117, 114, 99, 101),
      ::capnp::word(80, 97, 116, 104, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(24, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(153, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(152, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(164, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(161, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(160, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(172, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 253, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(169, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(164, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(176, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 252, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(173, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(172, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(184, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 251, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(181, 0, 0, 0, 210, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(188, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(200, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 250, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(197, 0, 0, 0, 146, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(200, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(212, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(102, 105, 108, 101, 83, 121, 115, 116),
      ::capnp::word(101, 109, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(110, 97, 109, 101, 115, 112, 97, 99),
      ::capnp::word(101, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 101, 109, 111, 114, 121, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(98, 117, 110, 100, 108, 101, 100, 84),
      ::capnp::word(121, 112, 101, 115, 104, 101, 100, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(98, 117, 110, 100, 108, 101, 100, 84),
      ::capnp::word(121, 112, 101, 115, 104, 101, 100, 84),
      ::capnp::word(104, 105, 114, 100, 80, 97, 114, 116),
      ::capnp::word(121, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(98, 117, 110, 100, 108, 101, 100, 84),
      ::capnp::word(104, 105, 114, 100, 80, 97, 114, 116),
      ::capnp::word(121, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        5 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1,2,3,4,5];
    pub static MEMBERS_BY_NAME : &[u16] = &[5,3,4,0,2,1];
    pub const TYPE_ID: u64 = 0xc978_9635_145f_607d;
  }
  pub enum Which<A0,A1,A2,A3,A4,A5> {
    FileSystem(A0),
    Namespace(A1),
    Memory(A2),
    BundledTypeshed(A3),
    BundledTypeshedThirdParty(A4),
    BundledThirdParty(A5),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<::capnp::text::Reader<'a>>,::capnp::Result<::capnp::text::Reader<'a>>,::capnp::Result<::capnp::text::Reader<'a>>,::capnp::Result<::capnp::text::Reader<'a>>,::capnp::Result<::capnp::text::Reader<'a>>,::capnp::Result<::capnp::text::Reader<'a>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<::capnp::text::Builder<'a>>,::capnp::Result<::capnp::text::Builder<'a>>,::capnp::Result<::capnp::text::Builder<'a>>,::capnp::Result<::capnp::text::Builder<'a>>,::capnp::Result<::capnp::text::Builder<'a>>,::capnp::Result<::capnp::text::Builder<'a>>>;
}

pub mod pysa_location {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <> Reader<'_,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_line(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_col(self) -> u32 {
      self.reader.get_data_field::<u32>(1)
    }
    #[inline]
    pub fn get_end_line(self) -> u32 {
      self.reader.get_data_field::<u32>(2)
    }
    #[inline]
    pub fn get_end_col(self) -> u32 {
      self.reader.get_data_field::<u32>(3)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 2, pointers: 0 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_line(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_line(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_col(self) -> u32 {
      self.builder.get_data_field::<u32>(1)
    }
    #[inline]
    pub fn set_col(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(1, value);
    }
    #[inline]
    pub fn get_end_line(self) -> u32 {
      self.builder.get_data_field::<u32>(2)
    }
    #[inline]
    pub fn set_end_line(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(2, value);
    }
    #[inline]
    pub fn get_end_col(self) -> u32 {
      self.builder.get_data_field::<u32>(3)
    }
    #[inline]
    pub fn set_end_col(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(3, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 81] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(42, 0, 0, 0, 1, 0, 2, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(0, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 186, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 231, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 121, 115, 97, 76, 111),
      ::capnp::word(99, 97, 116, 105, 111, 110, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(92, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(104, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 0, 0, 0, 34, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(112, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(116, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(108, 105, 110, 101, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 111, 108, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 110, 100, 76, 105, 110, 101, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 110, 100, 67, 111, 108, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        2 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        3 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,3,2,0];
    pub const TYPE_ID: u64 = 0xc309_c969_f3e5_8a60;
  }
}

pub mod class_ref {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <> Reader<'_,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_class_id(self) -> u32 {
      self.reader.get_data_field::<u32>(1)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 0 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_class_id(self) -> u32 {
      self.builder.get_data_field::<u32>(1)
    }
    #[inline]
    pub fn set_class_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(1, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 52] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(0, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 154, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 108, 97, 115, 115, 82),
      ::capnp::word(101, 102, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 108, 97, 115, 115, 73, 100, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0xde5e_9ff6_d0dd_25c7;
  }
}

pub mod function_ref {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_function_id(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_function_id(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 53] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(187, 184, 6, 16, 179, 91, 17, 128),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 178, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 82, 101, 102, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(48, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(60, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(73, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0x8011_5bb3_1006_b8bb;
  }
}

pub mod global_variable_ref {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 53] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(0, 136, 187, 143, 192, 52, 7, 247),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 71, 108, 111, 98, 97, 108),
      ::capnp::word(86, 97, 114, 105, 97, 98, 108, 101),
      ::capnp::word(82, 101, 102, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xf707_34c0_8fbb_8800;
  }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeModifier {
  Optional = 0,
  Coroutine = 1,
  Awaitable = 2,
  TypeVariableBound = 3,
  TypeVariableConstraint = 4,
  Type = 5,
}

impl ::capnp::introspect::Introspect for TypeModifier {
  fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Enum(::capnp::introspect::RawEnumSchema { encoded_node: &type_modifier::ENCODED_NODE, annotation_types: type_modifier::get_annotation_types }).into() }
}
impl ::core::convert::From<TypeModifier> for ::capnp::dynamic_value::Reader<'_> {
  fn from(e: TypeModifier) -> Self { ::capnp::dynamic_value::Enum::new(e.into(), ::capnp::introspect::RawEnumSchema { encoded_node: &type_modifier::ENCODED_NODE, annotation_types: type_modifier::get_annotation_types }.into()).into() }
}
impl ::core::convert::TryFrom<u16> for TypeModifier {
  type Error = ::capnp::NotInSchema;
  fn try_from(value: u16) -> ::core::result::Result<Self, <TypeModifier as ::core::convert::TryFrom<u16>>::Error> {
    match value {
      0 => ::core::result::Result::Ok(Self::Optional),
      1 => ::core::result::Result::Ok(Self::Coroutine),
      2 => ::core::result::Result::Ok(Self::Awaitable),
      3 => ::core::result::Result::Ok(Self::TypeVariableBound),
      4 => ::core::result::Result::Ok(Self::TypeVariableConstraint),
      5 => ::core::result::Result::Ok(Self::Type),
      n => ::core::result::Result::Err(::capnp::NotInSchema(n)),
    }
  }
}
impl From<TypeModifier> for u16 {
  #[inline]
  fn from(x: TypeModifier) -> u16 { x as u16 }
}
impl ::capnp::traits::HasTypeId for TypeModifier {
  const TYPE_ID: u64 = 0xf32f_ef75_955d_b263u64;
}
mod type_modifier {
pub static ENCODED_NODE: [::capnp::Word; 52] = [
  ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
  ::capnp::word(99, 178, 93, 149, 117, 239, 47, 243),
  ::capnp::word(42, 0, 0, 0, 2, 0, 0, 0),
  ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(21, 0, 0, 0, 186, 1, 0, 0),
  ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(41, 0, 0, 0, 151, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
  ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
  ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
  ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
  ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
  ::capnp::word(112, 58, 84, 121, 112, 101, 77, 111),
  ::capnp::word(100, 105, 102, 105, 101, 114, 0, 0),
  ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
  ::capnp::word(24, 0, 0, 0, 1, 0, 2, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(65, 0, 0, 0, 74, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(61, 0, 0, 0, 82, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(57, 0, 0, 0, 82, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(3, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(53, 0, 0, 0, 146, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(4, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(53, 0, 0, 0, 186, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(5, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(53, 0, 0, 0, 42, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(111, 112, 116, 105, 111, 110, 97, 108),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(99, 111, 114, 111, 117, 116, 105, 110),
  ::capnp::word(101, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(97, 119, 97, 105, 116, 97, 98, 108),
  ::capnp::word(101, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(116, 121, 112, 101, 86, 97, 114, 105),
  ::capnp::word(97, 98, 108, 101, 66, 111, 117, 110),
  ::capnp::word(100, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(116, 121, 112, 101, 86, 97, 114, 105),
  ::capnp::word(97, 98, 108, 101, 67, 111, 110, 115),
  ::capnp::word(116, 114, 97, 105, 110, 116, 0, 0),
  ::capnp::word(116, 121, 112, 101, 0, 0, 0, 0),
];
pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
  ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
}
}

pub mod class_with_modifiers {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_class(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_modifiers(self) -> ::capnp::Result<::capnp::enum_list::Reader<'a,crate::pysa_report_capnp::TypeModifier>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_modifiers(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_class(&mut self, value: crate::pysa_report_capnp::class_ref::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_class(self, ) -> crate::pysa_report_capnp::class_ref::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_class(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_modifiers(self) -> ::capnp::Result<::capnp::enum_list::Builder<'a,crate::pysa_report_capnp::TypeModifier>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_modifiers(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::enum_list::Owned<crate::pysa_report_capnp::TypeModifier>>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_modifiers(self, size: u32) -> ::capnp::enum_list::Builder<'a,crate::pysa_report_capnp::TypeModifier> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_modifiers(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_class(&self) -> crate::pysa_report_capnp::class_ref::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 57] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(54, 66, 125, 227, 173, 130, 95, 225),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 234, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 108, 97, 115, 115, 87),
      ::capnp::word(105, 116, 104, 77, 111, 100, 105, 102),
      ::capnp::word(105, 101, 114, 115, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(48, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(99, 108, 97, 115, 115, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 111, 100, 105, 102, 105, 101, 114),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 178, 93, 149, 117, 239, 47, 243),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::class_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::enum_list::Owned<crate::pysa_report_capnp::TypeModifier> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xe15f_82ad_e37d_4236;
  }
}

pub mod class_names_from_type {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_classes(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_with_modifiers::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_classes(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_is_exhaustive(self) -> bool {
      self.reader.get_bool_field(0)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_classes(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_with_modifiers::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_classes(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_with_modifiers::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_classes(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_with_modifiers::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_classes(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_is_exhaustive(self) -> bool {
      self.builder.get_bool_field(0)
    }
    #[inline]
    pub fn set_is_exhaustive(&mut self, value: bool)  {
      self.builder.set_bool_field(0, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 57] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(60, 232, 151, 46, 172, 74, 88, 160),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 234, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 108, 97, 115, 115, 78),
      ::capnp::word(97, 109, 101, 115, 70, 114, 111, 109),
      ::capnp::word(84, 121, 112, 101, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(64, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(61, 0, 0, 0, 106, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(60, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(99, 108, 97, 115, 115, 101, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(54, 66, 125, 227, 173, 130, 95, 225),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 69, 120, 104, 97, 117, 115),
      ::capnp::word(116, 105, 118, 101, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_with_modifiers::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <bool as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xa058_4aac_2e97_e83c;
  }
}

pub mod scalar_type_properties {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <> Reader<'_,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_is_bool(self) -> bool {
      self.reader.get_bool_field(0)
    }
    #[inline]
    pub fn get_is_int(self) -> bool {
      self.reader.get_bool_field(1)
    }
    #[inline]
    pub fn get_is_float(self) -> bool {
      self.reader.get_bool_field(2)
    }
    #[inline]
    pub fn get_is_enum(self) -> bool {
      self.reader.get_bool_field(3)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 0 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_is_bool(self) -> bool {
      self.builder.get_bool_field(0)
    }
    #[inline]
    pub fn set_is_bool(&mut self, value: bool)  {
      self.builder.set_bool_field(0, value);
    }
    #[inline]
    pub fn get_is_int(self) -> bool {
      self.builder.get_bool_field(1)
    }
    #[inline]
    pub fn set_is_int(&mut self, value: bool)  {
      self.builder.set_bool_field(1, value);
    }
    #[inline]
    pub fn get_is_float(self) -> bool {
      self.builder.get_bool_field(2)
    }
    #[inline]
    pub fn set_is_float(&mut self, value: bool)  {
      self.builder.set_bool_field(2, value);
    }
    #[inline]
    pub fn get_is_enum(self) -> bool {
      self.builder.get_bool_field(3)
    }
    #[inline]
    pub fn set_is_enum(&mut self, value: bool)  {
      self.builder.set_bool_field(3, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 82] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(191, 186, 161, 171, 141, 206, 154, 215),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(0, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 250, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 231, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 83, 99, 97, 108, 97, 114),
      ::capnp::word(84, 121, 112, 101, 80, 114, 111, 112),
      ::capnp::word(101, 114, 116, 105, 101, 115, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(92, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(104, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 0, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(112, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(116, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(105, 115, 66, 111, 111, 108, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 73, 110, 116, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 70, 108, 111, 97, 116, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 69, 110, 117, 109, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <bool as ::capnp::introspect::Introspect>::introspect(),
        1 => <bool as ::capnp::introspect::Introspect>::introspect(),
        2 => <bool as ::capnp::introspect::Introspect>::introspect(),
        3 => <bool as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,3,2,1];
    pub const TYPE_ID: u64 = 0xd79a_ce8d_aba1_babf;
  }
}

pub mod pysa_type {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_string(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_string(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_scalar_type_properties(self) -> ::capnp::Result<crate::pysa_report_capnp::scalar_type_properties::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_scalar_type_properties(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_class_names(self) -> ::capnp::Result<crate::pysa_report_capnp::class_names_from_type::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_class_names(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_string(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_string(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_string(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_string(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_scalar_type_properties(self) -> ::capnp::Result<crate::pysa_report_capnp::scalar_type_properties::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_scalar_type_properties(&mut self, value: crate::pysa_report_capnp::scalar_type_properties::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_scalar_type_properties(self, ) -> crate::pysa_report_capnp::scalar_type_properties::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_scalar_type_properties(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_class_names(self) -> ::capnp::Result<crate::pysa_report_capnp::class_names_from_type::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_class_names(&mut self, value: crate::pysa_report_capnp::class_names_from_type::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_class_names(self, ) -> crate::pysa_report_capnp::class_names_from_type::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), 0)
    }
    #[inline]
    pub fn has_class_names(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_scalar_type_properties(&self) -> crate::pysa_report_capnp::scalar_type_properties::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_class_names(&self) -> crate::pysa_report_capnp::class_names_from_type::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(2))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 69] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 154, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 121, 115, 97, 84, 121),
      ::capnp::word(112, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(76, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(73, 0, 0, 0, 170, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(76, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(88, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(85, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(84, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(96, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(115, 116, 114, 105, 110, 103, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(115, 99, 97, 108, 97, 114, 84, 121),
      ::capnp::word(112, 101, 80, 114, 111, 112, 101, 114),
      ::capnp::word(116, 105, 101, 115, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(191, 186, 161, 171, 141, 206, 154, 215),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 108, 97, 115, 115, 78, 97, 109),
      ::capnp::word(101, 115, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(60, 232, 151, 46, 172, 74, 88, 160),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::scalar_type_properties::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::class_names_from_type::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[2,1,0];
    pub const TYPE_ID: u64 = 0xd28d_95e8_ed28_2312;
  }
}

pub mod scope_parent {
  pub use self::Which::{Function,Class,TopLevel};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_function(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_class(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 1 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Function(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Class(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(TopLevel(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_function(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_function(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_function(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_class(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_class(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_class(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 1 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_top_level(&mut self, _value: ())  {
      self.builder.set_data_field::<u16>(0, 2);
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Function(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Class(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(TopLevel(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 68] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(142, 70, 87, 231, 224, 99, 36, 222),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 3, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 178, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 83, 99, 111, 112, 101, 80),
      ::capnp::word(97, 114, 101, 110, 116, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 0, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(72, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(84, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 253, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(81, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(80, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(92, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 108, 97, 115, 115, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 111, 112, 76, 101, 118, 101, 108),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <() as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0,2];
    pub const TYPE_ID: u64 = 0xde24_63e0_e757_468e;
  }
  pub enum Which<A0,A1> {
    Function(A0),
    Class(A1),
    TopLevel(()),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>>>;
}

pub mod function_parameter {
  pub use self::Which::{PosOnly,Pos,VarArg,KwOnly,Kwargs};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_pos_only(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_pos(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 1 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_var_arg(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 2 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_kw_only(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 3 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_kwargs(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 4 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(PosOnly(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Pos(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(VarArg(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        3 => {
          ::core::result::Result::Ok(KwOnly(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        4 => {
          ::core::result::Result::Ok(Kwargs(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_pos_only(&mut self, value: crate::pysa_report_capnp::function_parameter::pos_only_param::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_pos_only(self, ) -> crate::pysa_report_capnp::function_parameter::pos_only_param::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_pos_only(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_pos(&mut self, value: crate::pysa_report_capnp::function_parameter::pos_param::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_pos(self, ) -> crate::pysa_report_capnp::function_parameter::pos_param::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_pos(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 1 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_var_arg(&mut self, value: crate::pysa_report_capnp::function_parameter::var_arg_param::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 2);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_var_arg(self, ) -> crate::pysa_report_capnp::function_parameter::var_arg_param::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 2);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_var_arg(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 2 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_kw_only(&mut self, value: crate::pysa_report_capnp::function_parameter::kw_only_param::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 3);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_kw_only(self, ) -> crate::pysa_report_capnp::function_parameter::kw_only_param::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 3);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_kw_only(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 3 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_kwargs(&mut self, value: crate::pysa_report_capnp::function_parameter::kwargs_param::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 4);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_kwargs(self, ) -> crate::pysa_report_capnp::function_parameter::kwargs_param::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 4);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_kwargs(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 4 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(PosOnly(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Pos(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(VarArg(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        3 => {
          ::core::result::Result::Ok(KwOnly(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        4 => {
          ::core::result::Result::Ok(Kwargs(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 117] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 5, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 87, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 0, 0, 0, 31, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
      ::capnp::word(116, 101, 114, 0, 0, 0, 0, 0),
      ::capnp::word(20, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(86, 154, 151, 233, 120, 112, 75, 244),
      ::capnp::word(33, 0, 0, 0, 106, 0, 0, 0),
      ::capnp::word(207, 204, 169, 82, 198, 246, 5, 203),
      ::capnp::word(33, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(87, 92, 41, 152, 252, 244, 127, 152),
      ::capnp::word(33, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(105, 20, 192, 36, 216, 246, 239, 156),
      ::capnp::word(33, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(35, 177, 234, 49, 255, 49, 71, 145),
      ::capnp::word(33, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(80, 111, 115, 79, 110, 108, 121, 80),
      ::capnp::word(97, 114, 97, 109, 0, 0, 0, 0),
      ::capnp::word(80, 111, 115, 80, 97, 114, 97, 109),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(86, 97, 114, 65, 114, 103, 80, 97),
      ::capnp::word(114, 97, 109, 0, 0, 0, 0, 0),
      ::capnp::word(75, 119, 79, 110, 108, 121, 80, 97),
      ::capnp::word(114, 97, 109, 0, 0, 0, 0, 0),
      ::capnp::word(75, 119, 97, 114, 103, 115, 80, 97),
      ::capnp::word(114, 97, 109, 0, 0, 0, 0, 0),
      ::capnp::word(20, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(120, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(132, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(129, 0, 0, 0, 34, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(124, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(136, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 253, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(133, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(128, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(140, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 252, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(137, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(132, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(144, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 251, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(141, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(136, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(148, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(112, 111, 115, 79, 110, 108, 121, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(86, 154, 151, 233, 120, 112, 75, 244),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 111, 115, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(207, 204, 169, 82, 198, 246, 5, 203),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(118, 97, 114, 65, 114, 103, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(87, 92, 41, 152, 252, 244, 127, 152),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(107, 119, 79, 110, 108, 121, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 20, 192, 36, 216, 246, 239, 156),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(107, 119, 97, 114, 103, 115, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(35, 177, 234, 49, 255, 49, 71, 145),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::function_parameter::pos_only_param::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::function_parameter::pos_param::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::function_parameter::var_arg_param::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <crate::pysa_report_capnp::function_parameter::kw_only_param::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <crate::pysa_report_capnp::function_parameter::kwargs_param::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1,2,3,4];
    pub static MEMBERS_BY_NAME : &[u16] = &[3,4,1,0,2];
    pub const TYPE_ID: u64 = 0xa1b2_fce9_8392_735b;
  }
  pub enum Which<A0,A1,A2,A3,A4> {
    PosOnly(A0),
    Pos(A1),
    VarArg(A2),
    KwOnly(A3),
    Kwargs(A4),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::function_parameter::pos_only_param::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::pos_param::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::var_arg_param::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::kw_only_param::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::kwargs_param::Reader<'a>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::function_parameter::pos_only_param::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::pos_param::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::var_arg_param::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::kw_only_param::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::function_parameter::kwargs_param::Builder<'a>>>;

  pub mod pos_only_param {
    #[derive(Copy, Clone)]
    pub struct Owned(());
    impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
    impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

    pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
    impl <> ::core::marker::Copy for Reader<'_,>  {}
    impl <> ::core::clone::Clone for Reader<'_,>  {
      fn clone(&self) -> Self { *self }
    }

    impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
      fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
        Self { reader,  }
      }
    }

    impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
      fn from(reader: Reader<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <> ::core::fmt::Debug for Reader<'_,>  {
      fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
        core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
      }
    }

    impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
      fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(reader.get_struct(default)?.into())
      }
    }

    impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
      fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
        self.reader
      }
    }

    impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
      fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
        self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
      }
    }

    impl <'a,> Reader<'a,>  {
      pub fn reborrow(&self) -> Reader<'_,> {
        Self { .. *self }
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.reader.total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.reader.get_pointer_field(0).is_null()
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.reader.get_pointer_field(1).is_null()
      }
      #[inline]
      pub fn get_required(self) -> bool {
        self.reader.get_bool_field(0)
      }
    }

    pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
    impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
      const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 2 };
    }
    impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
      fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
        Self { builder,  }
      }
    }

    impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
      fn from(builder: Builder<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
      fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
        self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
      }
    }

    impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
      fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
        builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
      }
      fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
      }
    }

    impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
      fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
    }

    impl <'a,> Builder<'a,>  {
      pub fn into_reader(self) -> Reader<'a,> {
        self.builder.into_reader().into()
      }
      pub fn reborrow(&mut self) -> Builder<'_,> {
        Builder { builder: self.builder.reborrow() }
      }
      pub fn reborrow_as_reader(&self) -> Reader<'_,> {
        self.builder.as_reader().into()
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.builder.as_reader().total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
      }
      #[inline]
      pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
        self.builder.get_pointer_field(0).init_text(size)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.builder.is_pointer_field_null(0)
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_annotation(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
      }
      #[inline]
      pub fn init_annotation(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
        ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.builder.is_pointer_field_null(1)
      }
      #[inline]
      pub fn get_required(self) -> bool {
        self.builder.get_bool_field(0)
      }
      #[inline]
      pub fn set_required(&mut self, value: bool)  {
        self.builder.set_bool_field(0, value);
      }
    }

    pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
    impl ::capnp::capability::FromTypelessPipeline for Pipeline {
      fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
        Self { _typeless: typeless,  }
      }
    }
    impl Pipeline  {
      pub fn get_annotation(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
        ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
      }
    }
    mod _private {
      pub static ENCODED_NODE: [::capnp::Word; 71] = [
        ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
        ::capnp::word(86, 154, 151, 233, 120, 112, 75, 244),
        ::capnp::word(60, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
        ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(21, 0, 0, 0, 74, 2, 0, 0),
        ::capnp::word(57, 0, 0, 0, 7, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(53, 0, 0, 0, 175, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
        ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
        ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
        ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
        ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
        ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
        ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
        ::capnp::word(116, 101, 114, 46, 80, 111, 115, 79),
        ::capnp::word(110, 108, 121, 80, 97, 114, 97, 109),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(69, 0, 0, 0, 42, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(76, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(73, 0, 0, 0, 90, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(72, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(84, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(81, 0, 0, 0, 74, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(80, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(92, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(97, 110, 110, 111, 116, 97, 116, 105),
        ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(114, 101, 113, 117, 105, 114, 101, 100),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ];
      pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
        match index {
          0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
          1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
          2 => <bool as ::capnp::introspect::Introspect>::introspect(),
          _ => ::capnp::introspect::panic_invalid_field_index(index),
        }
      }
      pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
        ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
      }
      pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
      pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
        &ARENA,
        NONUNION_MEMBERS,
        MEMBERS_BY_DISCRIMINANT,
        MEMBERS_BY_NAME
      );
      pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
      pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
      pub static MEMBERS_BY_NAME : &[u16] = &[1,0,2];
      pub const TYPE_ID: u64 = 0xf44b_7078_e997_9a56;
    }
  }

  pub mod pos_param {
    #[derive(Copy, Clone)]
    pub struct Owned(());
    impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
    impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

    pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
    impl <> ::core::marker::Copy for Reader<'_,>  {}
    impl <> ::core::clone::Clone for Reader<'_,>  {
      fn clone(&self) -> Self { *self }
    }

    impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
      fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
        Self { reader,  }
      }
    }

    impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
      fn from(reader: Reader<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <> ::core::fmt::Debug for Reader<'_,>  {
      fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
        core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
      }
    }

    impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
      fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(reader.get_struct(default)?.into())
      }
    }

    impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
      fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
        self.reader
      }
    }

    impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
      fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
        self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
      }
    }

    impl <'a,> Reader<'a,>  {
      pub fn reborrow(&self) -> Reader<'_,> {
        Self { .. *self }
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.reader.total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.reader.get_pointer_field(0).is_null()
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.reader.get_pointer_field(1).is_null()
      }
      #[inline]
      pub fn get_required(self) -> bool {
        self.reader.get_bool_field(0)
      }
    }

    pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
    impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
      const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 2 };
    }
    impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
      fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
        Self { builder,  }
      }
    }

    impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
      fn from(builder: Builder<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
      fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
        self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
      }
    }

    impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
      fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
        builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
      }
      fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
      }
    }

    impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
      fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
    }

    impl <'a,> Builder<'a,>  {
      pub fn into_reader(self) -> Reader<'a,> {
        self.builder.into_reader().into()
      }
      pub fn reborrow(&mut self) -> Builder<'_,> {
        Builder { builder: self.builder.reborrow() }
      }
      pub fn reborrow_as_reader(&self) -> Reader<'_,> {
        self.builder.as_reader().into()
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.builder.as_reader().total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
      }
      #[inline]
      pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
        self.builder.get_pointer_field(0).init_text(size)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.builder.is_pointer_field_null(0)
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_annotation(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
      }
      #[inline]
      pub fn init_annotation(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
        ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.builder.is_pointer_field_null(1)
      }
      #[inline]
      pub fn get_required(self) -> bool {
        self.builder.get_bool_field(0)
      }
      #[inline]
      pub fn set_required(&mut self, value: bool)  {
        self.builder.set_bool_field(0, value);
      }
    }

    pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
    impl ::capnp::capability::FromTypelessPipeline for Pipeline {
      fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
        Self { _typeless: typeless,  }
      }
    }
    impl Pipeline  {
      pub fn get_annotation(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
        ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
      }
    }
    mod _private {
      pub static ENCODED_NODE: [::capnp::Word; 70] = [
        ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
        ::capnp::word(207, 204, 169, 82, 198, 246, 5, 203),
        ::capnp::word(60, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
        ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(21, 0, 0, 0, 42, 2, 0, 0),
        ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(49, 0, 0, 0, 175, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
        ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
        ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
        ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
        ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
        ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
        ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
        ::capnp::word(116, 101, 114, 46, 80, 111, 115, 80),
        ::capnp::word(97, 114, 97, 109, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(69, 0, 0, 0, 42, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(76, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(73, 0, 0, 0, 90, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(72, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(84, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(81, 0, 0, 0, 74, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(80, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(92, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(97, 110, 110, 111, 116, 97, 116, 105),
        ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(114, 101, 113, 117, 105, 114, 101, 100),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ];
      pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
        match index {
          0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
          1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
          2 => <bool as ::capnp::introspect::Introspect>::introspect(),
          _ => ::capnp::introspect::panic_invalid_field_index(index),
        }
      }
      pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
        ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
      }
      pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
      pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
        &ARENA,
        NONUNION_MEMBERS,
        MEMBERS_BY_DISCRIMINANT,
        MEMBERS_BY_NAME
      );
      pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
      pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
      pub static MEMBERS_BY_NAME : &[u16] = &[1,0,2];
      pub const TYPE_ID: u64 = 0xcb05_f6c6_52a9_cccf;
    }
  }

  pub mod var_arg_param {
    #[derive(Copy, Clone)]
    pub struct Owned(());
    impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
    impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

    pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
    impl <> ::core::marker::Copy for Reader<'_,>  {}
    impl <> ::core::clone::Clone for Reader<'_,>  {
      fn clone(&self) -> Self { *self }
    }

    impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
      fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
        Self { reader,  }
      }
    }

    impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
      fn from(reader: Reader<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <> ::core::fmt::Debug for Reader<'_,>  {
      fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
        core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
      }
    }

    impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
      fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(reader.get_struct(default)?.into())
      }
    }

    impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
      fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
        self.reader
      }
    }

    impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
      fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
        self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
      }
    }

    impl <'a,> Reader<'a,>  {
      pub fn reborrow(&self) -> Reader<'_,> {
        Self { .. *self }
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.reader.total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.reader.get_pointer_field(0).is_null()
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.reader.get_pointer_field(1).is_null()
      }
    }

    pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
    impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
      const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
    }
    impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
      fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
        Self { builder,  }
      }
    }

    impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
      fn from(builder: Builder<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
      fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
        self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
      }
    }

    impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
      fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
        builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
      }
      fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
      }
    }

    impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
      fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
    }

    impl <'a,> Builder<'a,>  {
      pub fn into_reader(self) -> Reader<'a,> {
        self.builder.into_reader().into()
      }
      pub fn reborrow(&mut self) -> Builder<'_,> {
        Builder { builder: self.builder.reborrow() }
      }
      pub fn reborrow_as_reader(&self) -> Reader<'_,> {
        self.builder.as_reader().into()
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.builder.as_reader().total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
      }
      #[inline]
      pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
        self.builder.get_pointer_field(0).init_text(size)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.builder.is_pointer_field_null(0)
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_annotation(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
      }
      #[inline]
      pub fn init_annotation(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
        ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.builder.is_pointer_field_null(1)
      }
    }

    pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
    impl ::capnp::capability::FromTypelessPipeline for Pipeline {
      fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
        Self { _typeless: typeless,  }
      }
    }
    impl Pipeline  {
      pub fn get_annotation(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
        ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
      }
    }
    mod _private {
      pub static ENCODED_NODE: [::capnp::Word; 54] = [
        ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
        ::capnp::word(87, 92, 41, 152, 252, 244, 127, 152),
        ::capnp::word(60, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
        ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(21, 0, 0, 0, 66, 2, 0, 0),
        ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(49, 0, 0, 0, 119, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
        ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
        ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
        ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
        ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
        ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
        ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
        ::capnp::word(116, 101, 114, 46, 86, 97, 114, 65),
        ::capnp::word(114, 103, 80, 97, 114, 97, 109, 0),
        ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(41, 0, 0, 0, 42, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(48, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(45, 0, 0, 0, 90, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(97, 110, 110, 111, 116, 97, 116, 105),
        ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ];
      pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
        match index {
          0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
          1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
          _ => ::capnp::introspect::panic_invalid_field_index(index),
        }
      }
      pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
        ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
      }
      pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
      pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
        &ARENA,
        NONUNION_MEMBERS,
        MEMBERS_BY_DISCRIMINANT,
        MEMBERS_BY_NAME
      );
      pub static NONUNION_MEMBERS : &[u16] = &[0,1];
      pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
      pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
      pub const TYPE_ID: u64 = 0x987f_f4fc_9829_5c57;
    }
  }

  pub mod kw_only_param {
    #[derive(Copy, Clone)]
    pub struct Owned(());
    impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
    impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

    pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
    impl <> ::core::marker::Copy for Reader<'_,>  {}
    impl <> ::core::clone::Clone for Reader<'_,>  {
      fn clone(&self) -> Self { *self }
    }

    impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
      fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
        Self { reader,  }
      }
    }

    impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
      fn from(reader: Reader<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <> ::core::fmt::Debug for Reader<'_,>  {
      fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
        core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
      }
    }

    impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
      fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(reader.get_struct(default)?.into())
      }
    }

    impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
      fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
        self.reader
      }
    }

    impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
      fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
        self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
      }
    }

    impl <'a,> Reader<'a,>  {
      pub fn reborrow(&self) -> Reader<'_,> {
        Self { .. *self }
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.reader.total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.reader.get_pointer_field(0).is_null()
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.reader.get_pointer_field(1).is_null()
      }
      #[inline]
      pub fn get_required(self) -> bool {
        self.reader.get_bool_field(0)
      }
    }

    pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
    impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
      const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 2 };
    }
    impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
      fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
        Self { builder,  }
      }
    }

    impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
      fn from(builder: Builder<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
      fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
        self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
      }
    }

    impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
      fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
        builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
      }
      fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
      }
    }

    impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
      fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
    }

    impl <'a,> Builder<'a,>  {
      pub fn into_reader(self) -> Reader<'a,> {
        self.builder.into_reader().into()
      }
      pub fn reborrow(&mut self) -> Builder<'_,> {
        Builder { builder: self.builder.reborrow() }
      }
      pub fn reborrow_as_reader(&self) -> Reader<'_,> {
        self.builder.as_reader().into()
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.builder.as_reader().total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
      }
      #[inline]
      pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
        self.builder.get_pointer_field(0).init_text(size)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.builder.is_pointer_field_null(0)
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_annotation(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
      }
      #[inline]
      pub fn init_annotation(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
        ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.builder.is_pointer_field_null(1)
      }
      #[inline]
      pub fn get_required(self) -> bool {
        self.builder.get_bool_field(0)
      }
      #[inline]
      pub fn set_required(&mut self, value: bool)  {
        self.builder.set_bool_field(0, value);
      }
    }

    pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
    impl ::capnp::capability::FromTypelessPipeline for Pipeline {
      fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
        Self { _typeless: typeless,  }
      }
    }
    impl Pipeline  {
      pub fn get_annotation(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
        ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
      }
    }
    mod _private {
      pub static ENCODED_NODE: [::capnp::Word; 70] = [
        ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
        ::capnp::word(105, 20, 192, 36, 216, 246, 239, 156),
        ::capnp::word(60, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
        ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(21, 0, 0, 0, 66, 2, 0, 0),
        ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(49, 0, 0, 0, 175, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
        ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
        ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
        ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
        ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
        ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
        ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
        ::capnp::word(116, 101, 114, 46, 75, 119, 79, 110),
        ::capnp::word(108, 121, 80, 97, 114, 97, 109, 0),
        ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(69, 0, 0, 0, 42, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(76, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(73, 0, 0, 0, 90, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(72, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(84, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(81, 0, 0, 0, 74, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(80, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(92, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(97, 110, 110, 111, 116, 97, 116, 105),
        ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(114, 101, 113, 117, 105, 114, 101, 100),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ];
      pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
        match index {
          0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
          1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
          2 => <bool as ::capnp::introspect::Introspect>::introspect(),
          _ => ::capnp::introspect::panic_invalid_field_index(index),
        }
      }
      pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
        ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
      }
      pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
      pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
        &ARENA,
        NONUNION_MEMBERS,
        MEMBERS_BY_DISCRIMINANT,
        MEMBERS_BY_NAME
      );
      pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
      pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
      pub static MEMBERS_BY_NAME : &[u16] = &[1,0,2];
      pub const TYPE_ID: u64 = 0x9cef_f6d8_24c0_1469;
    }
  }

  pub mod kwargs_param {
    #[derive(Copy, Clone)]
    pub struct Owned(());
    impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
    impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
    impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

    pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
    impl <> ::core::marker::Copy for Reader<'_,>  {}
    impl <> ::core::clone::Clone for Reader<'_,>  {
      fn clone(&self) -> Self { *self }
    }

    impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
      fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
        Self { reader,  }
      }
    }

    impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
      fn from(reader: Reader<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <> ::core::fmt::Debug for Reader<'_,>  {
      fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
        core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
      }
    }

    impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
      fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(reader.get_struct(default)?.into())
      }
    }

    impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
      fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
        self.reader
      }
    }

    impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
      fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
        self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
      }
    }

    impl <'a,> Reader<'a,>  {
      pub fn reborrow(&self) -> Reader<'_,> {
        Self { .. *self }
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.reader.total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.reader.get_pointer_field(0).is_null()
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
        ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.reader.get_pointer_field(1).is_null()
      }
    }

    pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
    impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
      const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
    }
    impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
      const TYPE_ID: u64 = _private::TYPE_ID;
    }
    impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
      fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
        Self { builder,  }
      }
    }

    impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
      fn from(builder: Builder<'a,>) -> Self {
        Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
      }
    }

    impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
      fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
        self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
      }
    }

    impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
      fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
        builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
      }
      fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
        ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
      }
    }

    impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
      fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
    }

    impl <'a,> Builder<'a,>  {
      pub fn into_reader(self) -> Reader<'a,> {
        self.builder.into_reader().into()
      }
      pub fn reborrow(&mut self) -> Builder<'_,> {
        Builder { builder: self.builder.reborrow() }
      }
      pub fn reborrow_as_reader(&self) -> Reader<'_,> {
        self.builder.as_reader().into()
      }

      pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
        self.builder.as_reader().total_size()
      }
      #[inline]
      pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
      }
      #[inline]
      pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
        self.builder.get_pointer_field(0).init_text(size)
      }
      #[inline]
      pub fn has_name(&self) -> bool {
        !self.builder.is_pointer_field_null(0)
      }
      #[inline]
      pub fn get_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
        ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
      }
      #[inline]
      pub fn set_annotation(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
        ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
      }
      #[inline]
      pub fn init_annotation(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
        ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
      }
      #[inline]
      pub fn has_annotation(&self) -> bool {
        !self.builder.is_pointer_field_null(1)
      }
    }

    pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
    impl ::capnp::capability::FromTypelessPipeline for Pipeline {
      fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
        Self { _typeless: typeless,  }
      }
    }
    impl Pipeline  {
      pub fn get_annotation(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
        ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
      }
    }
    mod _private {
      pub static ENCODED_NODE: [::capnp::Word; 54] = [
        ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
        ::capnp::word(35, 177, 234, 49, 255, 49, 71, 145),
        ::capnp::word(60, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
        ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(21, 0, 0, 0, 66, 2, 0, 0),
        ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(49, 0, 0, 0, 119, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
        ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
        ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
        ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
        ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
        ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
        ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
        ::capnp::word(116, 101, 114, 46, 75, 119, 97, 114),
        ::capnp::word(103, 115, 80, 97, 114, 97, 109, 0),
        ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
        ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(41, 0, 0, 0, 42, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(48, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(45, 0, 0, 0, 90, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
        ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
        ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(97, 110, 110, 111, 116, 97, 116, 105),
        ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
        ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ];
      pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
        match index {
          0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
          1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
          _ => ::capnp::introspect::panic_invalid_field_index(index),
        }
      }
      pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
        ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
      }
      pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
      pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
        &ARENA,
        NONUNION_MEMBERS,
        MEMBERS_BY_DISCRIMINANT,
        MEMBERS_BY_NAME
      );
      pub static NONUNION_MEMBERS : &[u16] = &[0,1];
      pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
      pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
      pub const TYPE_ID: u64 = 0x9147_31ff_31ea_b123;
    }
  }
}

pub mod function_parameters {
  pub use self::Which::{List,Ellipsis,ParamSpec};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_list(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(List(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Ellipsis(
            ()
          ))
        }
        2 => {
          ::core::result::Result::Ok(ParamSpec(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_list(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::function_parameter::Owned>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_list(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_parameter::Owned> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_list(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_ellipsis(&mut self, _value: ())  {
      self.builder.set_data_field::<u16>(0, 1);
    }
    #[inline]
    pub fn set_param_spec(&mut self, _value: ())  {
      self.builder.set_data_field::<u16>(0, 2);
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(List(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Ellipsis(
            ()
          ))
        }
        2 => {
          ::core::result::Result::Ok(ParamSpec(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 73] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(200, 112, 166, 14, 97, 163, 198, 167),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 3, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 234, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 80, 97, 114, 97, 109, 101),
      ::capnp::word(116, 101, 114, 115, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(92, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(89, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(88, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(100, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 253, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(108, 105, 115, 116, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(91, 115, 146, 131, 233, 252, 178, 161),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 108, 108, 105, 112, 115, 105, 115),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 97, 114, 97, 109, 83, 112, 101),
      ::capnp::word(99, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::function_parameter::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <() as ::capnp::introspect::Introspect>::introspect(),
        2 => <() as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0,2];
    pub const TYPE_ID: u64 = 0xa7c6_a361_0ea6_70c8;
  }
  pub enum Which<A0> {
    List(A0),
    Ellipsis(()),
    ParamSpec(()),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::function_parameter::Owned>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_parameter::Owned>>>;
}

pub mod function_signature {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_parameters(self) -> ::capnp::Result<crate::pysa_report_capnp::function_parameters::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_parameters(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_return_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_return_annotation(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_parameters(self) -> ::capnp::Result<crate::pysa_report_capnp::function_parameters::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_parameters(&mut self, value: crate::pysa_report_capnp::function_parameters::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_parameters(self, ) -> crate::pysa_report_capnp::function_parameters::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_parameters(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_return_annotation(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_return_annotation(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_return_annotation(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_return_annotation(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_parameters(&self) -> crate::pysa_report_capnp::function_parameters::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
    pub fn get_return_annotation(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 55] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(217, 183, 126, 139, 3, 218, 17, 190),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 83, 105, 103, 110, 97, 116),
      ::capnp::word(117, 114, 101, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(52, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(64, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(112, 97, 114, 97, 109, 101, 116, 101),
      ::capnp::word(114, 115, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(200, 112, 166, 14, 97, 163, 198, 167),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(114, 101, 116, 117, 114, 110, 65, 110),
      ::capnp::word(110, 111, 116, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::function_parameters::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xbe11_da03_8b7e_b7d9;
  }
}

pub mod function_base_definition {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_parent(self) -> ::capnp::Result<crate::pysa_report_capnp::scope_parent::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_parent(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_is_overload(self) -> bool {
      self.reader.get_bool_field(0)
    }
    #[inline]
    pub fn get_is_staticmethod(self) -> bool {
      self.reader.get_bool_field(1)
    }
    #[inline]
    pub fn get_is_classmethod(self) -> bool {
      self.reader.get_bool_field(2)
    }
    #[inline]
    pub fn get_is_property_getter(self) -> bool {
      self.reader.get_bool_field(3)
    }
    #[inline]
    pub fn get_is_property_setter(self) -> bool {
      self.reader.get_bool_field(4)
    }
    #[inline]
    pub fn get_is_stub(self) -> bool {
      self.reader.get_bool_field(5)
    }
    #[inline]
    pub fn get_is_def_statement(self) -> bool {
      self.reader.get_bool_field(6)
    }
    #[inline]
    pub fn get_defining_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_defining_class(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_parent(self) -> ::capnp::Result<crate::pysa_report_capnp::scope_parent::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_parent(&mut self, value: crate::pysa_report_capnp::scope_parent::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_parent(self, ) -> crate::pysa_report_capnp::scope_parent::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_parent(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_is_overload(self) -> bool {
      self.builder.get_bool_field(0)
    }
    #[inline]
    pub fn set_is_overload(&mut self, value: bool)  {
      self.builder.set_bool_field(0, value);
    }
    #[inline]
    pub fn get_is_staticmethod(self) -> bool {
      self.builder.get_bool_field(1)
    }
    #[inline]
    pub fn set_is_staticmethod(&mut self, value: bool)  {
      self.builder.set_bool_field(1, value);
    }
    #[inline]
    pub fn get_is_classmethod(self) -> bool {
      self.builder.get_bool_field(2)
    }
    #[inline]
    pub fn set_is_classmethod(&mut self, value: bool)  {
      self.builder.set_bool_field(2, value);
    }
    #[inline]
    pub fn get_is_property_getter(self) -> bool {
      self.builder.get_bool_field(3)
    }
    #[inline]
    pub fn set_is_property_getter(&mut self, value: bool)  {
      self.builder.set_bool_field(3, value);
    }
    #[inline]
    pub fn get_is_property_setter(self) -> bool {
      self.builder.get_bool_field(4)
    }
    #[inline]
    pub fn set_is_property_setter(&mut self, value: bool)  {
      self.builder.set_bool_field(4, value);
    }
    #[inline]
    pub fn get_is_stub(self) -> bool {
      self.builder.get_bool_field(5)
    }
    #[inline]
    pub fn set_is_stub(&mut self, value: bool)  {
      self.builder.set_bool_field(5, value);
    }
    #[inline]
    pub fn get_is_def_statement(self) -> bool {
      self.builder.get_bool_field(6)
    }
    #[inline]
    pub fn set_is_def_statement(&mut self, value: bool)  {
      self.builder.set_bool_field(6, value);
    }
    #[inline]
    pub fn get_defining_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_defining_class(&mut self, value: crate::pysa_report_capnp::class_ref::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_defining_class(self, ) -> crate::pysa_report_capnp::class_ref::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), 0)
    }
    #[inline]
    pub fn has_defining_class(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_parent(&self) -> crate::pysa_report_capnp::scope_parent::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_defining_class(&self) -> crate::pysa_report_capnp::class_ref::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(2))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 182] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(88, 55, 96, 28, 114, 220, 108, 135),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 10, 2, 0, 0),
      ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 55, 2, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 66, 97, 115, 101, 68, 101),
      ::capnp::word(102, 105, 110, 105, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(9, 1, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(4, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(13, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(20, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(17, 1, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(28, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(25, 1, 0, 0, 122, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(24, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(36, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(33, 1, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(32, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(44, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 1, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(6, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(53, 1, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(56, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(68, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(7, 0, 0, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(65, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(60, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 8, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 1, 0, 0, 122, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(9, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 9, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 1, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(76, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(88, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 97, 114, 101, 110, 116, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(142, 70, 87, 231, 224, 99, 36, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 79, 118, 101, 114, 108, 111),
      ::capnp::word(97, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 83, 116, 97, 116, 105, 99),
      ::capnp::word(109, 101, 116, 104, 111, 100, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 67, 108, 97, 115, 115, 109),
      ::capnp::word(101, 116, 104, 111, 100, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 80, 114, 111, 112, 101, 114),
      ::capnp::word(116, 121, 71, 101, 116, 116, 101, 114),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 80, 114, 111, 112, 101, 114),
      ::capnp::word(116, 121, 83, 101, 116, 116, 101, 114),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 83, 116, 117, 98, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 68, 101, 102, 83, 116, 97),
      ::capnp::word(116, 101, 109, 101, 110, 116, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 101, 102, 105, 110, 105, 110, 103),
      ::capnp::word(67, 108, 97, 115, 115, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::scope_parent::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <bool as ::capnp::introspect::Introspect>::introspect(),
        3 => <bool as ::capnp::introspect::Introspect>::introspect(),
        4 => <bool as ::capnp::introspect::Introspect>::introspect(),
        5 => <bool as ::capnp::introspect::Introspect>::introspect(),
        6 => <bool as ::capnp::introspect::Introspect>::introspect(),
        7 => <bool as ::capnp::introspect::Introspect>::introspect(),
        8 => <bool as ::capnp::introspect::Introspect>::introspect(),
        9 => <crate::pysa_report_capnp::class_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5,6,7,8,9];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[9,4,8,2,5,6,3,7,0,1];
    pub const TYPE_ID: u64 = 0x876c_dc72_1c60_3758;
  }
}

pub mod captured_variable_ref {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_outer_function(self) -> ::capnp::Result<crate::pysa_report_capnp::function_ref::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_outer_function(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_outer_function(self) -> ::capnp::Result<crate::pysa_report_capnp::function_ref::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_outer_function(&mut self, value: crate::pysa_report_capnp::function_ref::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_outer_function(self, ) -> crate::pysa_report_capnp::function_ref::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_outer_function(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(1).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_outer_function(&self) -> crate::pysa_report_capnp::function_ref::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 53] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(78, 187, 214, 48, 136, 94, 128, 148),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 242, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 97, 112, 116, 117, 114),
      ::capnp::word(101, 100, 86, 97, 114, 105, 97, 98),
      ::capnp::word(108, 101, 82, 101, 102, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(111, 117, 116, 101, 114, 70, 117, 110),
      ::capnp::word(99, 116, 105, 111, 110, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(187, 184, 6, 16, 179, 91, 17, 128),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::function_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0x9480_5e88_30d6_bb4e;
  }
}

pub mod target {
  pub use self::Which::{Function,Overrides,FormatString};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_function(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_overrides(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 1 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Function(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Overrides(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(FormatString(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_function(&mut self, value: crate::pysa_report_capnp::function_ref::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_function(self, ) -> crate::pysa_report_capnp::function_ref::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_function(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_overrides(&mut self, value: crate::pysa_report_capnp::function_ref::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_overrides(self, ) -> crate::pysa_report_capnp::function_ref::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_overrides(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 1 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_format_string(&mut self, _value: ())  {
      self.builder.set_data_field::<u16>(0, 2);
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Function(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Overrides(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(FormatString(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 69] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(211, 25, 219, 52, 204, 102, 130, 255),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 3, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 138, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 84, 97, 114, 103, 101, 116),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(76, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(88, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 253, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(85, 0, 0, 0, 106, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(84, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(96, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(187, 184, 6, 16, 179, 91, 17, 128),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(111, 118, 101, 114, 114, 105, 100, 101),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(187, 184, 6, 16, 179, 91, 17, 128),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 111, 114, 109, 97, 116, 83, 116),
      ::capnp::word(114, 105, 110, 103, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::function_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::function_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <() as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_NAME : &[u16] = &[2,0,1];
    pub const TYPE_ID: u64 = 0xff82_66cc_34db_19d3;
  }
  pub enum Which<A0,A1> {
    Function(A0),
    Overrides(A1),
    FormatString(()),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::function_ref::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::function_ref::Reader<'a>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::function_ref::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::function_ref::Builder<'a>>>;
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplicitReceiver {
  TrueWithClassReceiver = 0,
  TrueWithObjectReceiver = 1,
  False = 2,
}

impl ::capnp::introspect::Introspect for ImplicitReceiver {
  fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Enum(::capnp::introspect::RawEnumSchema { encoded_node: &implicit_receiver::ENCODED_NODE, annotation_types: implicit_receiver::get_annotation_types }).into() }
}
impl ::core::convert::From<ImplicitReceiver> for ::capnp::dynamic_value::Reader<'_> {
  fn from(e: ImplicitReceiver) -> Self { ::capnp::dynamic_value::Enum::new(e.into(), ::capnp::introspect::RawEnumSchema { encoded_node: &implicit_receiver::ENCODED_NODE, annotation_types: implicit_receiver::get_annotation_types }.into()).into() }
}
impl ::core::convert::TryFrom<u16> for ImplicitReceiver {
  type Error = ::capnp::NotInSchema;
  fn try_from(value: u16) -> ::core::result::Result<Self, <ImplicitReceiver as ::core::convert::TryFrom<u16>>::Error> {
    match value {
      0 => ::core::result::Result::Ok(Self::TrueWithClassReceiver),
      1 => ::core::result::Result::Ok(Self::TrueWithObjectReceiver),
      2 => ::core::result::Result::Ok(Self::False),
      n => ::core::result::Result::Err(::capnp::NotInSchema(n)),
    }
  }
}
impl From<ImplicitReceiver> for u16 {
  #[inline]
  fn from(x: ImplicitReceiver) -> u16 { x as u16 }
}
impl ::capnp::traits::HasTypeId for ImplicitReceiver {
  const TYPE_ID: u64 = 0xe411_df5e_f616_6ad0u64;
}
mod implicit_receiver {
pub static ENCODED_NODE: [::capnp::Word; 38] = [
  ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
  ::capnp::word(208, 106, 22, 246, 94, 223, 17, 228),
  ::capnp::word(42, 0, 0, 0, 2, 0, 0, 0),
  ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(21, 0, 0, 0, 218, 1, 0, 0),
  ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(45, 0, 0, 0, 79, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
  ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
  ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
  ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
  ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
  ::capnp::word(112, 58, 73, 109, 112, 108, 105, 99),
  ::capnp::word(105, 116, 82, 101, 99, 101, 105, 118),
  ::capnp::word(101, 114, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
  ::capnp::word(12, 0, 0, 0, 1, 0, 2, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(29, 0, 0, 0, 178, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(29, 0, 0, 0, 186, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(29, 0, 0, 0, 50, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(116, 114, 117, 101, 87, 105, 116, 104),
  ::capnp::word(67, 108, 97, 115, 115, 82, 101, 99),
  ::capnp::word(101, 105, 118, 101, 114, 0, 0, 0),
  ::capnp::word(116, 114, 117, 101, 87, 105, 116, 104),
  ::capnp::word(79, 98, 106, 101, 99, 116, 82, 101),
  ::capnp::word(99, 101, 105, 118, 101, 114, 0, 0),
  ::capnp::word(102, 97, 108, 115, 101, 0, 0, 0),
];
pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
  ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
}
}

pub mod pysa_call_target {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_target(self) -> ::capnp::Result<crate::pysa_report_capnp::target::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_target(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_implicit_receiver(self) -> ::core::result::Result<crate::pysa_report_capnp::ImplicitReceiver,::capnp::NotInSchema> {
      ::core::convert::TryInto::try_into(self.reader.get_data_field::<u16>(0))
    }
    #[inline]
    pub fn get_implicit_dunder_call(self) -> bool {
      self.reader.get_bool_field(16)
    }
    #[inline]
    pub fn get_receiver_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_receiver_class(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_is_class_method(self) -> bool {
      self.reader.get_bool_field(17)
    }
    #[inline]
    pub fn get_is_static_method(self) -> bool {
      self.reader.get_bool_field(18)
    }
    #[inline]
    pub fn get_return_type(self) -> ::capnp::Result<crate::pysa_report_capnp::scalar_type_properties::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_return_type(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_target(self) -> ::capnp::Result<crate::pysa_report_capnp::target::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_target(&mut self, value: crate::pysa_report_capnp::target::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_target(self, ) -> crate::pysa_report_capnp::target::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_target(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_implicit_receiver(self) -> ::core::result::Result<crate::pysa_report_capnp::ImplicitReceiver,::capnp::NotInSchema> {
      ::core::convert::TryInto::try_into(self.builder.get_data_field::<u16>(0))
    }
    #[inline]
    pub fn set_implicit_receiver(&mut self, value: crate::pysa_report_capnp::ImplicitReceiver)  {
      self.builder.set_data_field::<u16>(0, value as u16);
    }
    #[inline]
    pub fn get_implicit_dunder_call(self) -> bool {
      self.builder.get_bool_field(16)
    }
    #[inline]
    pub fn set_implicit_dunder_call(&mut self, value: bool)  {
      self.builder.set_bool_field(16, value);
    }
    #[inline]
    pub fn get_receiver_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_receiver_class(&mut self, value: crate::pysa_report_capnp::class_ref::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_receiver_class(self, ) -> crate::pysa_report_capnp::class_ref::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_receiver_class(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_is_class_method(self) -> bool {
      self.builder.get_bool_field(17)
    }
    #[inline]
    pub fn set_is_class_method(&mut self, value: bool)  {
      self.builder.set_bool_field(17, value);
    }
    #[inline]
    pub fn get_is_static_method(self) -> bool {
      self.builder.get_bool_field(18)
    }
    #[inline]
    pub fn set_is_static_method(&mut self, value: bool)  {
      self.builder.set_bool_field(18, value);
    }
    #[inline]
    pub fn get_return_type(self) -> ::capnp::Result<crate::pysa_report_capnp::scalar_type_properties::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_return_type(&mut self, value: crate::pysa_report_capnp::scalar_type_properties::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_return_type(self, ) -> crate::pysa_report_capnp::scalar_type_properties::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), 0)
    }
    #[inline]
    pub fn has_return_type(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_target(&self) -> crate::pysa_report_capnp::target::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
    pub fn get_receiver_class(&self) -> crate::pysa_report_capnp::class_ref::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_return_type(&self) -> crate::pysa_report_capnp::scalar_type_properties::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(2))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 135] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 202, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 143, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 121, 115, 97, 67, 97),
      ::capnp::word(108, 108, 84, 97, 114, 103, 101, 116),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(28, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(181, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(176, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(188, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(185, 0, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(188, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(200, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 16, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(197, 0, 0, 0, 154, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(200, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(212, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(209, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(208, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(220, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 17, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(217, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(216, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(228, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 18, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 0, 0, 0, 122, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(224, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(236, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(6, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(233, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(232, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(244, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(116, 97, 114, 103, 101, 116, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(211, 25, 219, 52, 204, 102, 130, 255),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 109, 112, 108, 105, 99, 105, 116),
      ::capnp::word(82, 101, 99, 101, 105, 118, 101, 114),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(208, 106, 22, 246, 94, 223, 17, 228),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 109, 112, 108, 105, 99, 105, 116),
      ::capnp::word(68, 117, 110, 100, 101, 114, 67, 97),
      ::capnp::word(108, 108, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(114, 101, 99, 101, 105, 118, 101, 114),
      ::capnp::word(67, 108, 97, 115, 115, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 67, 108, 97, 115, 115, 77),
      ::capnp::word(101, 116, 104, 111, 100, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 83, 116, 97, 116, 105, 99),
      ::capnp::word(77, 101, 116, 104, 111, 100, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(114, 101, 116, 117, 114, 110, 84, 121),
      ::capnp::word(112, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(191, 186, 161, 171, 141, 206, 154, 215),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::target::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::ImplicitReceiver as ::capnp::introspect::Introspect>::introspect(),
        2 => <bool as ::capnp::introspect::Introspect>::introspect(),
        3 => <crate::pysa_report_capnp::class_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <bool as ::capnp::introspect::Introspect>::introspect(),
        5 => <bool as ::capnp::introspect::Introspect>::introspect(),
        6 => <crate::pysa_report_capnp::scalar_type_properties::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5,6];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[2,1,4,5,3,6,0];
    pub const TYPE_ID: u64 = 0xcf15_42da_5a3f_29e1;
  }
}

pub mod decorator_callee {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_location(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_location(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_location(&self) -> crate::pysa_report_capnp::pysa_location::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 57] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(23, 16, 185, 122, 140, 62, 8, 188),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 210, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 68, 101, 99, 111, 114, 97),
      ::capnp::word(116, 111, 114, 67, 97, 108, 108, 101),
      ::capnp::word(101, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 97, 114, 103, 101, 116, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(211, 25, 219, 52, 204, 102, 130, 255),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xbc08_3e8c_7ab9_1017;
  }
}

pub mod function_definition {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_parent(self) -> ::capnp::Result<crate::pysa_report_capnp::scope_parent::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_parent(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_is_overload(self) -> bool {
      self.reader.get_bool_field(0)
    }
    #[inline]
    pub fn get_is_staticmethod(self) -> bool {
      self.reader.get_bool_field(1)
    }
    #[inline]
    pub fn get_is_classmethod(self) -> bool {
      self.reader.get_bool_field(2)
    }
    #[inline]
    pub fn get_is_property_getter(self) -> bool {
      self.reader.get_bool_field(3)
    }
    #[inline]
    pub fn get_is_property_setter(self) -> bool {
      self.reader.get_bool_field(4)
    }
    #[inline]
    pub fn get_is_stub(self) -> bool {
      self.reader.get_bool_field(5)
    }
    #[inline]
    pub fn get_is_def_statement(self) -> bool {
      self.reader.get_bool_field(6)
    }
    #[inline]
    pub fn get_defining_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_defining_class(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_undecorated_signatures(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::function_signature::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_undecorated_signatures(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
    #[inline]
    pub fn get_captured_variables(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::captured_variable_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_captured_variables(&self) -> bool {
      !self.reader.get_pointer_field(5).is_null()
    }
    #[inline]
    pub fn get_decorator_callees(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::decorator_callee::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(6), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_decorator_callees(&self) -> bool {
      !self.reader.get_pointer_field(6).is_null()
    }
    #[inline]
    pub fn get_overridden_base_method(self) -> ::capnp::Result<crate::pysa_report_capnp::function_ref::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(7), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_overridden_base_method(&self) -> bool {
      !self.reader.get_pointer_field(7).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 8 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_parent(self) -> ::capnp::Result<crate::pysa_report_capnp::scope_parent::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_parent(&mut self, value: crate::pysa_report_capnp::scope_parent::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_parent(self, ) -> crate::pysa_report_capnp::scope_parent::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_parent(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_is_overload(self) -> bool {
      self.builder.get_bool_field(0)
    }
    #[inline]
    pub fn set_is_overload(&mut self, value: bool)  {
      self.builder.set_bool_field(0, value);
    }
    #[inline]
    pub fn get_is_staticmethod(self) -> bool {
      self.builder.get_bool_field(1)
    }
    #[inline]
    pub fn set_is_staticmethod(&mut self, value: bool)  {
      self.builder.set_bool_field(1, value);
    }
    #[inline]
    pub fn get_is_classmethod(self) -> bool {
      self.builder.get_bool_field(2)
    }
    #[inline]
    pub fn set_is_classmethod(&mut self, value: bool)  {
      self.builder.set_bool_field(2, value);
    }
    #[inline]
    pub fn get_is_property_getter(self) -> bool {
      self.builder.get_bool_field(3)
    }
    #[inline]
    pub fn set_is_property_getter(&mut self, value: bool)  {
      self.builder.set_bool_field(3, value);
    }
    #[inline]
    pub fn get_is_property_setter(self) -> bool {
      self.builder.get_bool_field(4)
    }
    #[inline]
    pub fn set_is_property_setter(&mut self, value: bool)  {
      self.builder.set_bool_field(4, value);
    }
    #[inline]
    pub fn get_is_stub(self) -> bool {
      self.builder.get_bool_field(5)
    }
    #[inline]
    pub fn set_is_stub(&mut self, value: bool)  {
      self.builder.set_bool_field(5, value);
    }
    #[inline]
    pub fn get_is_def_statement(self) -> bool {
      self.builder.get_bool_field(6)
    }
    #[inline]
    pub fn set_is_def_statement(&mut self, value: bool)  {
      self.builder.set_bool_field(6, value);
    }
    #[inline]
    pub fn get_defining_class(self) -> ::capnp::Result<crate::pysa_report_capnp::class_ref::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_defining_class(&mut self, value: crate::pysa_report_capnp::class_ref::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_defining_class(self, ) -> crate::pysa_report_capnp::class_ref::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), 0)
    }
    #[inline]
    pub fn has_defining_class(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_function_id(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false).unwrap()
    }
    #[inline]
    pub fn init_function_id(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(3).init_text(size)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_undecorated_signatures(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_signature::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_undecorated_signatures(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::function_signature::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false)
    }
    #[inline]
    pub fn init_undecorated_signatures(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_signature::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(4), size)
    }
    #[inline]
    pub fn has_undecorated_signatures(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
    #[inline]
    pub fn get_captured_variables(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::captured_variable_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_captured_variables(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::captured_variable_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(5), value, false)
    }
    #[inline]
    pub fn init_captured_variables(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::captured_variable_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(5), size)
    }
    #[inline]
    pub fn has_captured_variables(&self) -> bool {
      !self.builder.is_pointer_field_null(5)
    }
    #[inline]
    pub fn get_decorator_callees(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::decorator_callee::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(6), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_decorator_callees(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::decorator_callee::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(6), value, false)
    }
    #[inline]
    pub fn init_decorator_callees(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::decorator_callee::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(6), size)
    }
    #[inline]
    pub fn has_decorator_callees(&self) -> bool {
      !self.builder.is_pointer_field_null(6)
    }
    #[inline]
    pub fn get_overridden_base_method(self) -> ::capnp::Result<crate::pysa_report_capnp::function_ref::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(7), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_overridden_base_method(&mut self, value: crate::pysa_report_capnp::function_ref::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(7), value, false)
    }
    #[inline]
    pub fn init_overridden_base_method(self, ) -> crate::pysa_report_capnp::function_ref::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(7), 0)
    }
    #[inline]
    pub fn has_overridden_base_method(&self) -> bool {
      !self.builder.is_pointer_field_null(7)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_parent(&self) -> crate::pysa_report_capnp::scope_parent::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_defining_class(&self) -> crate::pysa_report_capnp::class_ref::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(2))
    }
    pub fn get_overridden_base_method(&self) -> crate::pysa_report_capnp::function_ref::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(7))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 277] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(17, 100, 67, 113, 216, 204, 43, 172),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(8, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 234, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 79, 3, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 68, 101, 102, 105, 110, 105),
      ::capnp::word(116, 105, 111, 110, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(60, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(149, 1, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(144, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(156, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(153, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(148, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(160, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(157, 1, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(156, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(168, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(165, 1, 0, 0, 122, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(164, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(176, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(173, 1, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(172, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(184, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(181, 1, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(184, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(196, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(6, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(193, 1, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(196, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(208, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(7, 0, 0, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(205, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(200, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(212, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 8, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(209, 1, 0, 0, 122, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(208, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(220, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(9, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 9, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(217, 1, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(216, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(228, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(10, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 10, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 1, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(224, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(236, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(11, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 11, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(233, 1, 0, 0, 178, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(236, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(8, 2, 0, 0, 2, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 12, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(5, 2, 0, 0, 146, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 2, 0, 0, 3, 0, 1, 0),
      ::capnp::word(36, 2, 0, 0, 2, 0, 1, 0),
      ::capnp::word(13, 0, 0, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 13, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(33, 2, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 2, 0, 0, 3, 0, 1, 0),
      ::capnp::word(64, 2, 0, 0, 2, 0, 1, 0),
      ::capnp::word(14, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 14, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(61, 2, 0, 0, 170, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 2, 0, 0, 3, 0, 1, 0),
      ::capnp::word(76, 2, 0, 0, 2, 0, 1, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 97, 114, 101, 110, 116, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(142, 70, 87, 231, 224, 99, 36, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 79, 118, 101, 114, 108, 111),
      ::capnp::word(97, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 83, 116, 97, 116, 105, 99),
      ::capnp::word(109, 101, 116, 104, 111, 100, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 67, 108, 97, 115, 115, 109),
      ::capnp::word(101, 116, 104, 111, 100, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 80, 114, 111, 112, 101, 114),
      ::capnp::word(116, 121, 71, 101, 116, 116, 101, 114),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 80, 114, 111, 112, 101, 114),
      ::capnp::word(116, 121, 83, 101, 116, 116, 101, 114),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 83, 116, 117, 98, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 68, 101, 102, 83, 116, 97),
      ::capnp::word(116, 101, 109, 101, 110, 116, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 101, 102, 105, 110, 105, 110, 103),
      ::capnp::word(67, 108, 97, 115, 115, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(73, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(117, 110, 100, 101, 99, 111, 114, 97),
      ::capnp::word(116, 101, 100, 83, 105, 103, 110, 97),
      ::capnp::word(116, 117, 114, 101, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(217, 183, 126, 139, 3, 218, 17, 190),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 97, 112, 116, 117, 114, 101, 100),
      ::capnp::word(86, 97, 114, 105, 97, 98, 108, 101),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(78, 187, 214, 48, 136, 94, 128, 148),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 101, 99, 111, 114, 97, 116, 111),
      ::capnp::word(114, 67, 97, 108, 108, 101, 101, 115),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(23, 16, 185, 122, 140, 62, 8, 188),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(111, 118, 101, 114, 114, 105, 100, 100),
      ::capnp::word(101, 110, 66, 97, 115, 101, 77, 101),
      ::capnp::word(116, 104, 111, 100, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(187, 184, 6, 16, 179, 91, 17, 128),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::scope_parent::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <bool as ::capnp::introspect::Introspect>::introspect(),
        3 => <bool as ::capnp::introspect::Introspect>::introspect(),
        4 => <bool as ::capnp::introspect::Introspect>::introspect(),
        5 => <bool as ::capnp::introspect::Introspect>::introspect(),
        6 => <bool as ::capnp::introspect::Introspect>::introspect(),
        7 => <bool as ::capnp::introspect::Introspect>::introspect(),
        8 => <bool as ::capnp::introspect::Introspect>::introspect(),
        9 => <crate::pysa_report_capnp::class_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        10 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        11 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::function_signature::Owned> as ::capnp::introspect::Introspect>::introspect(),
        12 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::captured_variable_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        13 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::decorator_callee::Owned> as ::capnp::introspect::Introspect>::introspect(),
        14 => <crate::pysa_report_capnp::function_ref::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[12,13,9,10,4,8,2,5,6,3,7,0,14,1,11];
    pub const TYPE_ID: u64 = 0xac2b_ccd8_7143_6411;
  }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PysaClassFieldDeclaration {
  None = 0,
  DeclaredByAnnotation = 1,
  DeclaredWithoutAnnotation = 2,
  AssignedInBody = 3,
  DefinedWithoutAssign = 4,
  DefinedInMethod = 5,
}

impl ::capnp::introspect::Introspect for PysaClassFieldDeclaration {
  fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Enum(::capnp::introspect::RawEnumSchema { encoded_node: &pysa_class_field_declaration::ENCODED_NODE, annotation_types: pysa_class_field_declaration::get_annotation_types }).into() }
}
impl ::core::convert::From<PysaClassFieldDeclaration> for ::capnp::dynamic_value::Reader<'_> {
  fn from(e: PysaClassFieldDeclaration) -> Self { ::capnp::dynamic_value::Enum::new(e.into(), ::capnp::introspect::RawEnumSchema { encoded_node: &pysa_class_field_declaration::ENCODED_NODE, annotation_types: pysa_class_field_declaration::get_annotation_types }.into()).into() }
}
impl ::core::convert::TryFrom<u16> for PysaClassFieldDeclaration {
  type Error = ::capnp::NotInSchema;
  fn try_from(value: u16) -> ::core::result::Result<Self, <PysaClassFieldDeclaration as ::core::convert::TryFrom<u16>>::Error> {
    match value {
      0 => ::core::result::Result::Ok(Self::None),
      1 => ::core::result::Result::Ok(Self::DeclaredByAnnotation),
      2 => ::core::result::Result::Ok(Self::DeclaredWithoutAnnotation),
      3 => ::core::result::Result::Ok(Self::AssignedInBody),
      4 => ::core::result::Result::Ok(Self::DefinedWithoutAssign),
      5 => ::core::result::Result::Ok(Self::DefinedInMethod),
      n => ::core::result::Result::Err(::capnp::NotInSchema(n)),
    }
  }
}
impl From<PysaClassFieldDeclaration> for u16 {
  #[inline]
  fn from(x: PysaClassFieldDeclaration) -> u16 { x as u16 }
}
impl ::capnp::traits::HasTypeId for PysaClassFieldDeclaration {
  const TYPE_ID: u64 = 0xedc4_394d_3301_28efu64;
}
mod pysa_class_field_declaration {
pub static ENCODED_NODE: [::capnp::Word; 56] = [
  ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
  ::capnp::word(239, 40, 1, 51, 77, 57, 196, 237),
  ::capnp::word(42, 0, 0, 0, 2, 0, 0, 0),
  ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(21, 0, 0, 0, 34, 2, 0, 0),
  ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(49, 0, 0, 0, 151, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
  ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
  ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
  ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
  ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
  ::capnp::word(112, 58, 80, 121, 115, 97, 67, 108),
  ::capnp::word(97, 115, 115, 70, 105, 101, 108, 100),
  ::capnp::word(68, 101, 99, 108, 97, 114, 97, 116),
  ::capnp::word(105, 111, 110, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
  ::capnp::word(24, 0, 0, 0, 1, 0, 2, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(65, 0, 0, 0, 42, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(57, 0, 0, 0, 170, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(57, 0, 0, 0, 210, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(3, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(61, 0, 0, 0, 122, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(4, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(57, 0, 0, 0, 170, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(5, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(57, 0, 0, 0, 130, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(110, 111, 110, 101, 0, 0, 0, 0),
  ::capnp::word(100, 101, 99, 108, 97, 114, 101, 100),
  ::capnp::word(66, 121, 65, 110, 110, 111, 116, 97),
  ::capnp::word(116, 105, 111, 110, 0, 0, 0, 0),
  ::capnp::word(100, 101, 99, 108, 97, 114, 101, 100),
  ::capnp::word(87, 105, 116, 104, 111, 117, 116, 65),
  ::capnp::word(110, 110, 111, 116, 97, 116, 105, 111),
  ::capnp::word(110, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(97, 115, 115, 105, 103, 110, 101, 100),
  ::capnp::word(73, 110, 66, 111, 100, 121, 0, 0),
  ::capnp::word(100, 101, 102, 105, 110, 101, 100, 87),
  ::capnp::word(105, 116, 104, 111, 117, 116, 65, 115),
  ::capnp::word(115, 105, 103, 110, 0, 0, 0, 0),
  ::capnp::word(100, 101, 102, 105, 110, 101, 100, 73),
  ::capnp::word(110, 77, 101, 116, 104, 111, 100, 0),
];
pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
  ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
}
}

pub mod pysa_class_field {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_type(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_type(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_explicit_annotation(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_explicit_annotation(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_declaration_kind(self) -> ::core::result::Result<crate::pysa_report_capnp::PysaClassFieldDeclaration,::capnp::NotInSchema> {
      ::core::convert::TryInto::try_into(self.reader.get_data_field::<u16>(0))
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 4 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_type(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_type(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_type(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_type(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_explicit_annotation(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_explicit_annotation(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false).unwrap()
    }
    #[inline]
    pub fn init_explicit_annotation(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(2).init_text(size)
    }
    #[inline]
    pub fn has_explicit_annotation(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_location(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false)
    }
    #[inline]
    pub fn init_location(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(3), 0)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_declaration_kind(self) -> ::core::result::Result<crate::pysa_report_capnp::PysaClassFieldDeclaration,::capnp::NotInSchema> {
      ::core::convert::TryInto::try_into(self.builder.get_data_field::<u16>(0))
    }
    #[inline]
    pub fn set_declaration_kind(&mut self, value: crate::pysa_report_capnp::PysaClassFieldDeclaration)  {
      self.builder.set_data_field::<u16>(0, value as u16);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_type(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_location(&self) -> crate::pysa_report_capnp::pysa_location::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(3))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 101] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(37, 94, 154, 170, 230, 192, 78, 249),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(4, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 202, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 31, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 121, 115, 97, 67, 108),
      ::capnp::word(97, 115, 115, 70, 105, 101, 108, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(20, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(120, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(132, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(129, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(124, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(136, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(133, 0, 0, 0, 154, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(136, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(148, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(145, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(144, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(156, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(153, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(152, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(164, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 121, 112, 101, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 120, 112, 108, 105, 99, 105, 116),
      ::capnp::word(65, 110, 110, 111, 116, 97, 116, 105),
      ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 101, 99, 108, 97, 114, 97, 116),
      ::capnp::word(105, 111, 110, 75, 105, 110, 100, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(239, 40, 1, 51, 77, 57, 196, 237),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <crate::pysa_report_capnp::PysaClassFieldDeclaration as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[4,2,3,0,1];
    pub const TYPE_ID: u64 = 0xf94e_c0e6_aa9a_5e25;
  }
}

pub mod pysa_class_mro {
  pub use self::Which::{Resolved,Cyclic};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_resolved(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Resolved(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Cyclic(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_resolved(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_ref::Owned>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_resolved(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_resolved(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_cyclic(&mut self, _value: ())  {
      self.builder.set_data_field::<u16>(0, 1);
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Resolved(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Cyclic(
            ()
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 56] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(251, 223, 38, 45, 197, 175, 218, 173),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 2, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 186, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 121, 115, 97, 67, 108),
      ::capnp::word(97, 115, 115, 77, 114, 111, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(68, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(65, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(60, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(114, 101, 115, 111, 108, 118, 101, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 121, 99, 108, 105, 99, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <() as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0xadda_afc5_2d26_dffb;
  }
  pub enum Which<A0> {
    Resolved(A0),
    Cyclic(()),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_ref::Owned>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned>>>;
}

pub mod class_definition {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_class_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_bases(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_bases(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_mro(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_class_mro::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_mro(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_parent(self) -> ::capnp::Result<crate::pysa_report_capnp::scope_parent::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_parent(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
    #[inline]
    pub fn get_is_synthesized(self) -> bool {
      self.reader.get_bool_field(32)
    }
    #[inline]
    pub fn get_is_dataclass(self) -> bool {
      self.reader.get_bool_field(33)
    }
    #[inline]
    pub fn get_is_named_tuple(self) -> bool {
      self.reader.get_bool_field(34)
    }
    #[inline]
    pub fn get_is_typed_dict(self) -> bool {
      self.reader.get_bool_field(35)
    }
    #[inline]
    pub fn get_fields(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_class_field::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_fields(&self) -> bool {
      !self.reader.get_pointer_field(5).is_null()
    }
    #[inline]
    pub fn get_decorator_callees(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::decorator_callee::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(6), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_decorator_callees(&self) -> bool {
      !self.reader.get_pointer_field(6).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 7 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_location(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_location(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_class_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_class_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(1).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_bases(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_bases(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_bases(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_bases(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_mro(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_class_mro::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_mro(&mut self, value: crate::pysa_report_capnp::pysa_class_mro::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false)
    }
    #[inline]
    pub fn init_mro(self, ) -> crate::pysa_report_capnp::pysa_class_mro::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(3), 0)
    }
    #[inline]
    pub fn has_mro(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_parent(self) -> ::capnp::Result<crate::pysa_report_capnp::scope_parent::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_parent(&mut self, value: crate::pysa_report_capnp::scope_parent::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false)
    }
    #[inline]
    pub fn init_parent(self, ) -> crate::pysa_report_capnp::scope_parent::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(4), 0)
    }
    #[inline]
    pub fn has_parent(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
    #[inline]
    pub fn get_is_synthesized(self) -> bool {
      self.builder.get_bool_field(32)
    }
    #[inline]
    pub fn set_is_synthesized(&mut self, value: bool)  {
      self.builder.set_bool_field(32, value);
    }
    #[inline]
    pub fn get_is_dataclass(self) -> bool {
      self.builder.get_bool_field(33)
    }
    #[inline]
    pub fn set_is_dataclass(&mut self, value: bool)  {
      self.builder.set_bool_field(33, value);
    }
    #[inline]
    pub fn get_is_named_tuple(self) -> bool {
      self.builder.get_bool_field(34)
    }
    #[inline]
    pub fn set_is_named_tuple(&mut self, value: bool)  {
      self.builder.set_bool_field(34, value);
    }
    #[inline]
    pub fn get_is_typed_dict(self) -> bool {
      self.builder.get_bool_field(35)
    }
    #[inline]
    pub fn set_is_typed_dict(&mut self, value: bool)  {
      self.builder.set_bool_field(35, value);
    }
    #[inline]
    pub fn get_fields(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_class_field::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_fields(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_class_field::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(5), value, false)
    }
    #[inline]
    pub fn init_fields(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_class_field::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(5), size)
    }
    #[inline]
    pub fn has_fields(&self) -> bool {
      !self.builder.is_pointer_field_null(5)
    }
    #[inline]
    pub fn get_decorator_callees(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::decorator_callee::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(6), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_decorator_callees(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::decorator_callee::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(6), value, false)
    }
    #[inline]
    pub fn init_decorator_callees(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::decorator_callee::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(6), size)
    }
    #[inline]
    pub fn has_decorator_callees(&self) -> bool {
      !self.builder.is_pointer_field_null(6)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_location(&self) -> crate::pysa_report_capnp::pysa_location::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
    pub fn get_mro(&self) -> crate::pysa_report_capnp::pysa_class_mro::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(3))
    }
    pub fn get_parent(&self) -> crate::pysa_report_capnp::scope_parent::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(4))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 221] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(25, 191, 82, 128, 248, 185, 2, 248),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(7, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 210, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 167, 2, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 108, 97, 115, 115, 68),
      ::capnp::word(101, 102, 105, 110, 105, 116, 105, 111),
      ::capnp::word(110, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(48, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(65, 1, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(76, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(73, 1, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 1, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(72, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(84, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(81, 1, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(76, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(104, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 1, 0, 0, 34, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(112, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(6, 0, 0, 0, 32, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 1, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(108, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(120, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(7, 0, 0, 0, 33, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(117, 1, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(128, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 34, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 8, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 1, 0, 0, 106, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(124, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(136, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(9, 0, 0, 0, 35, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 9, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(133, 1, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(132, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(144, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(10, 0, 0, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 10, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(141, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(136, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(164, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(11, 0, 0, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 11, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(161, 1, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(164, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(192, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 108, 97, 115, 115, 73, 100, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(98, 97, 115, 101, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 114, 111, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(251, 223, 38, 45, 197, 175, 218, 173),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 97, 114, 101, 110, 116, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(142, 70, 87, 231, 224, 99, 36, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 83, 121, 110, 116, 104, 101),
      ::capnp::word(115, 105, 122, 101, 100, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 68, 97, 116, 97, 99, 108),
      ::capnp::word(97, 115, 115, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 78, 97, 109, 101, 100, 84),
      ::capnp::word(117, 112, 108, 101, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 84, 121, 112, 101, 100, 68),
      ::capnp::word(105, 99, 116, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 105, 101, 108, 100, 115, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(37, 94, 154, 170, 230, 192, 78, 249),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 101, 99, 111, 114, 97, 116, 111),
      ::capnp::word(114, 67, 97, 108, 108, 101, 101, 115),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(23, 16, 185, 122, 140, 62, 8, 188),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        4 => <crate::pysa_report_capnp::pysa_class_mro::Owned as ::capnp::introspect::Introspect>::introspect(),
        5 => <crate::pysa_report_capnp::scope_parent::Owned as ::capnp::introspect::Introspect>::introspect(),
        6 => <bool as ::capnp::introspect::Introspect>::introspect(),
        7 => <bool as ::capnp::introspect::Introspect>::introspect(),
        8 => <bool as ::capnp::introspect::Introspect>::introspect(),
        9 => <bool as ::capnp::introspect::Introspect>::introspect(),
        10 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_class_field::Owned> as ::capnp::introspect::Introspect>::introspect(),
        11 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::decorator_callee::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5,6,7,8,9,10,11];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[3,1,11,10,7,8,6,9,0,4,2,5];
    pub const TYPE_ID: u64 = 0xf802_b9f8_8052_bf19;
  }
}

pub mod global_variable {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_type(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_type(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_type(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_type::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_type(&mut self, value: crate::pysa_report_capnp::pysa_type::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_type(self, ) -> crate::pysa_report_capnp::pysa_type::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_type(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_location(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_location(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), 0)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_type(&self) -> crate::pysa_report_capnp::pysa_type::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_location(&self) -> crate::pysa_report_capnp::pysa_location::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(2))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 68] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(107, 4, 36, 64, 15, 189, 236, 215),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 202, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 71, 108, 111, 98, 97, 108),
      ::capnp::word(86, 97, 114, 105, 97, 98, 108, 101),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(76, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(73, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(76, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(88, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 121, 112, 101, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::pysa_type::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[2,0,1];
    pub const TYPE_ID: u64 = 0xd7ec_bd0f_4024_046b;
  }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnresolvedReason {
  LambdaArgument = 0,
  UnexpectedPyreflyTarget = 1,
  EmptyPyreflyCallTarget = 2,
  UnknownClassField = 3,
  ClassFieldOnlyExistInObject = 4,
  UnsupportedFunctionTarget = 5,
  UnexpectedDefiningClass = 6,
  UnexpectedInitMethod = 7,
  UnexpectedNewMethod = 8,
  UnexpectedCalleeExpression = 9,
  UnresolvedMagicDunderAttr = 10,
  UnresolvedMagicDunderAttrDueToNoBase = 11,
  UnresolvedMagicDunderAttrDueToNoAttribute = 12,
  Mixed = 13,
}

impl ::capnp::introspect::Introspect for UnresolvedReason {
  fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Enum(::capnp::introspect::RawEnumSchema { encoded_node: &unresolved_reason::ENCODED_NODE, annotation_types: unresolved_reason::get_annotation_types }).into() }
}
impl ::core::convert::From<UnresolvedReason> for ::capnp::dynamic_value::Reader<'_> {
  fn from(e: UnresolvedReason) -> Self { ::capnp::dynamic_value::Enum::new(e.into(), ::capnp::introspect::RawEnumSchema { encoded_node: &unresolved_reason::ENCODED_NODE, annotation_types: unresolved_reason::get_annotation_types }.into()).into() }
}
impl ::core::convert::TryFrom<u16> for UnresolvedReason {
  type Error = ::capnp::NotInSchema;
  fn try_from(value: u16) -> ::core::result::Result<Self, <UnresolvedReason as ::core::convert::TryFrom<u16>>::Error> {
    match value {
      0 => ::core::result::Result::Ok(Self::LambdaArgument),
      1 => ::core::result::Result::Ok(Self::UnexpectedPyreflyTarget),
      2 => ::core::result::Result::Ok(Self::EmptyPyreflyCallTarget),
      3 => ::core::result::Result::Ok(Self::UnknownClassField),
      4 => ::core::result::Result::Ok(Self::ClassFieldOnlyExistInObject),
      5 => ::core::result::Result::Ok(Self::UnsupportedFunctionTarget),
      6 => ::core::result::Result::Ok(Self::UnexpectedDefiningClass),
      7 => ::core::result::Result::Ok(Self::UnexpectedInitMethod),
      8 => ::core::result::Result::Ok(Self::UnexpectedNewMethod),
      9 => ::core::result::Result::Ok(Self::UnexpectedCalleeExpression),
      10 => ::core::result::Result::Ok(Self::UnresolvedMagicDunderAttr),
      11 => ::core::result::Result::Ok(Self::UnresolvedMagicDunderAttrDueToNoBase),
      12 => ::core::result::Result::Ok(Self::UnresolvedMagicDunderAttrDueToNoAttribute),
      13 => ::core::result::Result::Ok(Self::Mixed),
      n => ::core::result::Result::Err(::capnp::NotInSchema(n)),
    }
  }
}
impl From<UnresolvedReason> for u16 {
  #[inline]
  fn from(x: UnresolvedReason) -> u16 { x as u16 }
}
impl ::capnp::traits::HasTypeId for UnresolvedReason {
  const TYPE_ID: u64 = 0xe19c_52e7_d7b6_4732u64;
}
mod unresolved_reason {
pub static ENCODED_NODE: [::capnp::Word; 112] = [
  ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
  ::capnp::word(50, 71, 182, 215, 231, 82, 156, 225),
  ::capnp::word(42, 0, 0, 0, 2, 0, 0, 0),
  ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(21, 0, 0, 0, 218, 1, 0, 0),
  ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(45, 0, 0, 0, 87, 1, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
  ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
  ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
  ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
  ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
  ::capnp::word(112, 58, 85, 110, 114, 101, 115, 111),
  ::capnp::word(108, 118, 101, 100, 82, 101, 97, 115),
  ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
  ::capnp::word(56, 0, 0, 0, 1, 0, 2, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(161, 0, 0, 0, 122, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(157, 0, 0, 0, 194, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(2, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(157, 0, 0, 0, 186, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(3, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(157, 0, 0, 0, 146, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(4, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(157, 0, 0, 0, 226, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(5, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(161, 0, 0, 0, 210, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(6, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(165, 0, 0, 0, 194, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(7, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(165, 0, 0, 0, 170, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(165, 0, 0, 0, 162, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(9, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(165, 0, 0, 0, 218, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(10, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(169, 0, 0, 0, 210, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(11, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(173, 0, 0, 0, 42, 1, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(181, 0, 0, 0, 82, 1, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(13, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(193, 0, 0, 0, 50, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(108, 97, 109, 98, 100, 97, 65, 114),
  ::capnp::word(103, 117, 109, 101, 110, 116, 0, 0),
  ::capnp::word(117, 110, 101, 120, 112, 101, 99, 116),
  ::capnp::word(101, 100, 80, 121, 114, 101, 102, 108),
  ::capnp::word(121, 84, 97, 114, 103, 101, 116, 0),
  ::capnp::word(101, 109, 112, 116, 121, 80, 121, 114),
  ::capnp::word(101, 102, 108, 121, 67, 97, 108, 108),
  ::capnp::word(84, 97, 114, 103, 101, 116, 0, 0),
  ::capnp::word(117, 110, 107, 110, 111, 119, 110, 67),
  ::capnp::word(108, 97, 115, 115, 70, 105, 101, 108),
  ::capnp::word(100, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(99, 108, 97, 115, 115, 70, 105, 101),
  ::capnp::word(108, 100, 79, 110, 108, 121, 69, 120),
  ::capnp::word(105, 115, 116, 73, 110, 79, 98, 106),
  ::capnp::word(101, 99, 116, 0, 0, 0, 0, 0),
  ::capnp::word(117, 110, 115, 117, 112, 112, 111, 114),
  ::capnp::word(116, 101, 100, 70, 117, 110, 99, 116),
  ::capnp::word(105, 111, 110, 84, 97, 114, 103, 101),
  ::capnp::word(116, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(117, 110, 101, 120, 112, 101, 99, 116),
  ::capnp::word(101, 100, 68, 101, 102, 105, 110, 105),
  ::capnp::word(110, 103, 67, 108, 97, 115, 115, 0),
  ::capnp::word(117, 110, 101, 120, 112, 101, 99, 116),
  ::capnp::word(101, 100, 73, 110, 105, 116, 77, 101),
  ::capnp::word(116, 104, 111, 100, 0, 0, 0, 0),
  ::capnp::word(117, 110, 101, 120, 112, 101, 99, 116),
  ::capnp::word(101, 100, 78, 101, 119, 77, 101, 116),
  ::capnp::word(104, 111, 100, 0, 0, 0, 0, 0),
  ::capnp::word(117, 110, 101, 120, 112, 101, 99, 116),
  ::capnp::word(101, 100, 67, 97, 108, 108, 101, 101),
  ::capnp::word(69, 120, 112, 114, 101, 115, 115, 105),
  ::capnp::word(111, 110, 0, 0, 0, 0, 0, 0),
  ::capnp::word(117, 110, 114, 101, 115, 111, 108, 118),
  ::capnp::word(101, 100, 77, 97, 103, 105, 99, 68),
  ::capnp::word(117, 110, 100, 101, 114, 65, 116, 116),
  ::capnp::word(114, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(117, 110, 114, 101, 115, 111, 108, 118),
  ::capnp::word(101, 100, 77, 97, 103, 105, 99, 68),
  ::capnp::word(117, 110, 100, 101, 114, 65, 116, 116),
  ::capnp::word(114, 68, 117, 101, 84, 111, 78, 111),
  ::capnp::word(66, 97, 115, 101, 0, 0, 0, 0),
  ::capnp::word(117, 110, 114, 101, 115, 111, 108, 118),
  ::capnp::word(101, 100, 77, 97, 103, 105, 99, 68),
  ::capnp::word(117, 110, 100, 101, 114, 65, 116, 116),
  ::capnp::word(114, 68, 117, 101, 84, 111, 78, 111),
  ::capnp::word(65, 116, 116, 114, 105, 98, 117, 116),
  ::capnp::word(101, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(109, 105, 120, 101, 100, 0, 0, 0),
];
pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
  ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
}
}

pub mod unresolved {
  pub use self::Which::{False,True};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <> Reader<'_,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(False(
            ()
          ))
        }
        1 => {
          ::core::result::Result::Ok(True(
            ::core::convert::TryInto::try_into(self.reader.get_data_field::<u16>(1))
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 0 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_false(&mut self, _value: ())  {
      self.builder.set_data_field::<u16>(0, 0);
    }
    #[inline]
    pub fn set_true(&mut self, value: crate::pysa_report_capnp::UnresolvedReason)  {
      self.builder.set_data_field::<u16>(0, 1);
      self.builder.set_data_field::<u16>(1, value as u16);
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(False(
            ()
          ))
        }
        1 => {
          ::core::result::Result::Ok(True(
            ::core::convert::TryInto::try_into(self.builder.get_data_field::<u16>(1))
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 51] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(179, 115, 129, 217, 190, 122, 190, 245),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(0, 0, 7, 0, 0, 0, 2, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 170, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 85, 110, 114, 101, 115, 111),
      ::capnp::word(108, 118, 101, 100, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(48, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(102, 97, 108, 115, 101, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 114, 117, 101, 0, 0, 0, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(50, 71, 182, 215, 231, 82, 156, 225),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <() as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::UnresolvedReason as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xf5be_7abe_d981_73b3;
  }
  pub enum Which {
    False(()),
    True(::core::result::Result<crate::pysa_report_capnp::UnresolvedReason,::capnp::NotInSchema>),
  }
  pub type WhichReader = Which;
  pub type WhichBuilder = Which;
}

pub mod higher_order_parameter {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_index(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_call_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_call_targets(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_unresolved(self) -> ::capnp::Result<crate::pysa_report_capnp::unresolved::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_unresolved(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_index(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_index(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_call_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_call_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_call_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_call_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_unresolved(self) -> ::capnp::Result<crate::pysa_report_capnp::unresolved::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_unresolved(&mut self, value: crate::pysa_report_capnp::unresolved::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_unresolved(self, ) -> crate::pysa_report_capnp::unresolved::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_unresolved(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_unresolved(&self) -> crate::pysa_report_capnp::unresolved::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 73] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(61, 185, 31, 242, 164, 242, 8, 159),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 250, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 72, 105, 103, 104, 101, 114),
      ::capnp::word(79, 114, 100, 101, 114, 80, 97, 114),
      ::capnp::word(97, 109, 101, 116, 101, 114, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(76, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(73, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(72, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(100, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(105, 110, 100, 101, 120, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 97, 108, 108, 84, 97, 114, 103),
      ::capnp::word(101, 116, 115, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(117, 110, 114, 101, 115, 111, 108, 118),
      ::capnp::word(101, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(179, 115, 129, 217, 190, 122, 190, 245),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::unresolved::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0,2];
    pub const TYPE_ID: u64 = 0x9f08_f2a4_f21f_b93d;
  }
}

pub mod call_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_call_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_call_targets(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_init_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_init_targets(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_new_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_new_targets(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_higher_order_parameters(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::higher_order_parameter::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_higher_order_parameters(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_unresolved(self) -> ::capnp::Result<crate::pysa_report_capnp::unresolved::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_unresolved(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 5 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_call_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_call_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_call_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_call_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_init_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_init_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_init_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_init_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_new_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_new_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_new_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_new_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_higher_order_parameters(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::higher_order_parameter::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_higher_order_parameters(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::higher_order_parameter::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false)
    }
    #[inline]
    pub fn init_higher_order_parameters(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::higher_order_parameter::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(3), size)
    }
    #[inline]
    pub fn has_higher_order_parameters(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_unresolved(self) -> ::capnp::Result<crate::pysa_report_capnp::unresolved::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_unresolved(&mut self, value: crate::pysa_report_capnp::unresolved::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false)
    }
    #[inline]
    pub fn init_unresolved(self, ) -> crate::pysa_report_capnp::unresolved::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(4), 0)
    }
    #[inline]
    pub fn has_unresolved(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_unresolved(&self) -> crate::pysa_report_capnp::unresolved::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(4))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 118] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(85, 150, 122, 198, 157, 194, 151, 250),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(5, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 178, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 31, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 97, 108, 108, 67, 97),
      ::capnp::word(108, 108, 101, 101, 115, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(20, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(124, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(152, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(149, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(148, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(176, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(173, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(172, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(200, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(197, 0, 0, 0, 178, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(200, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(228, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(224, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(236, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(99, 97, 108, 108, 84, 97, 114, 103),
      ::capnp::word(101, 116, 115, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 110, 105, 116, 84, 97, 114, 103),
      ::capnp::word(101, 116, 115, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(110, 101, 119, 84, 97, 114, 103, 101),
      ::capnp::word(116, 115, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 105, 103, 104, 101, 114, 79, 114),
      ::capnp::word(100, 101, 114, 80, 97, 114, 97, 109),
      ::capnp::word(101, 116, 101, 114, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(61, 185, 31, 242, 164, 242, 8, 159),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(117, 110, 114, 101, 115, 111, 108, 118),
      ::capnp::word(101, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(179, 115, 129, 217, 190, 122, 190, 245),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::higher_order_parameter::Owned> as ::capnp::introspect::Introspect>::introspect(),
        4 => <crate::pysa_report_capnp::unresolved::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,3,1,2,4];
    pub const TYPE_ID: u64 = 0xfa97_c29d_c67a_9655;
  }
}

pub mod attribute_access_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_if_called(self) -> ::capnp::Result<crate::pysa_report_capnp::call_callees::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_if_called(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_property_setters(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_property_setters(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_property_getters(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_property_getters(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_global_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::global_variable_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_global_targets(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_is_attribute(self) -> bool {
      self.reader.get_bool_field(0)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 4 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_if_called(self) -> ::capnp::Result<crate::pysa_report_capnp::call_callees::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_if_called(&mut self, value: crate::pysa_report_capnp::call_callees::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_if_called(self, ) -> crate::pysa_report_capnp::call_callees::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_if_called(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_property_setters(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_property_setters(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_property_setters(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_property_setters(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_property_getters(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_property_getters(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_property_getters(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_property_getters(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_global_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::global_variable_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_global_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::global_variable_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false)
    }
    #[inline]
    pub fn init_global_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::global_variable_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(3), size)
    }
    #[inline]
    pub fn has_global_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_is_attribute(self) -> bool {
      self.builder.get_bool_field(0)
    }
    #[inline]
    pub fn set_is_attribute(&mut self, value: bool)  {
      self.builder.set_bool_field(0, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_if_called(&self) -> crate::pysa_report_capnp::call_callees::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 115] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(1, 114, 216, 20, 247, 203, 3, 218),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(4, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 10, 2, 0, 0),
      ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 31, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 65, 116, 116, 114, 105, 98),
      ::capnp::word(117, 116, 101, 65, 99, 99, 101, 115),
      ::capnp::word(115, 67, 97, 108, 108, 101, 101, 115),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(20, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(124, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(136, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(133, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(132, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(160, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(157, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(156, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(184, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(181, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(180, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(208, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(205, 0, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(204, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(216, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(105, 102, 67, 97, 108, 108, 101, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(85, 150, 122, 198, 157, 194, 151, 250),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 114, 111, 112, 101, 114, 116, 121),
      ::capnp::word(83, 101, 116, 116, 101, 114, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 114, 111, 112, 101, 114, 116, 121),
      ::capnp::word(71, 101, 116, 116, 101, 114, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(103, 108, 111, 98, 97, 108, 84, 97),
      ::capnp::word(114, 103, 101, 116, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 136, 187, 143, 192, 52, 7, 247),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 65, 116, 116, 114, 105, 98),
      ::capnp::word(117, 116, 101, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::call_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::global_variable_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        4 => <bool as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[3,0,4,2,1];
    pub const TYPE_ID: u64 = 0xda03_cbf7_14d8_7201;
  }
}

pub mod identifier_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_if_called(self) -> ::capnp::Result<crate::pysa_report_capnp::call_callees::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_if_called(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_global_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::global_variable_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_global_targets(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_captured_variables(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::captured_variable_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_captured_variables(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_if_called(self) -> ::capnp::Result<crate::pysa_report_capnp::call_callees::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_if_called(&mut self, value: crate::pysa_report_capnp::call_callees::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_if_called(self, ) -> crate::pysa_report_capnp::call_callees::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_if_called(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_global_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::global_variable_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_global_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::global_variable_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_global_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::global_variable_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_global_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_captured_variables(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::captured_variable_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_captured_variables(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::captured_variable_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_captured_variables(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::captured_variable_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_captured_variables(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_if_called(&self) -> crate::pysa_report_capnp::call_callees::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 79] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(41, 95, 40, 104, 0, 220, 240, 165),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 73, 100, 101, 110, 116, 105),
      ::capnp::word(102, 105, 101, 114, 67, 97, 108, 108),
      ::capnp::word(101, 101, 115, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(76, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(104, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 0, 0, 0, 146, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(132, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(105, 102, 67, 97, 108, 108, 101, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(85, 150, 122, 198, 157, 194, 151, 250),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(103, 108, 111, 98, 97, 108, 84, 97),
      ::capnp::word(114, 103, 101, 116, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 136, 187, 143, 192, 52, 7, 247),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 97, 112, 116, 117, 114, 101, 100),
      ::capnp::word(86, 97, 114, 105, 97, 98, 108, 101),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(78, 187, 214, 48, 136, 94, 128, 148),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::call_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::global_variable_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::captured_variable_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[2,1,0];
    pub const TYPE_ID: u64 = 0xa5f0_dc00_6828_5f29;
  }
}

pub mod define_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_define_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_define_targets(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_define_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_define_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_define_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_define_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 41] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(231, 220, 255, 73, 22, 33, 121, 179),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 194, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 63, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 68, 101, 102, 105, 110, 101),
      ::capnp::word(67, 97, 108, 108, 101, 101, 115, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(13, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(40, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(100, 101, 102, 105, 110, 101, 84, 97),
      ::capnp::word(114, 103, 101, 116, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0];
    pub const TYPE_ID: u64 = 0xb379_2116_49ff_dce7;
  }
}

pub mod format_string_artificial_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 42] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(67, 149, 110, 38, 253, 142, 93, 160),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 66, 2, 0, 0),
      ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 63, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 111, 114, 109, 97, 116),
      ::capnp::word(83, 116, 114, 105, 110, 103, 65, 114),
      ::capnp::word(116, 105, 102, 105, 99, 105, 97, 108),
      ::capnp::word(67, 97, 108, 108, 101, 101, 115, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(13, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(36, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(116, 97, 114, 103, 101, 116, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0];
    pub const TYPE_ID: u64 = 0xa05d_8efd_266e_9543;
  }
}

pub mod format_string_stringify_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_unresolved(self) -> ::capnp::Result<crate::pysa_report_capnp::unresolved::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_unresolved(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_unresolved(self) -> ::capnp::Result<crate::pysa_report_capnp::unresolved::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_unresolved(&mut self, value: crate::pysa_report_capnp::unresolved::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_unresolved(self, ) -> crate::pysa_report_capnp::unresolved::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_unresolved(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_unresolved(&self) -> crate::pysa_report_capnp::unresolved::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 58] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(179, 166, 153, 76, 210, 20, 114, 197),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 58, 2, 0, 0),
      ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 111, 114, 109, 97, 116),
      ::capnp::word(83, 116, 114, 105, 110, 103, 83, 116),
      ::capnp::word(114, 105, 110, 103, 105, 102, 121, 67),
      ::capnp::word(97, 108, 108, 101, 101, 115, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(64, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(61, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(60, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(116, 97, 114, 103, 101, 116, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(117, 110, 114, 101, 115, 111, 108, 118),
      ::capnp::word(101, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(179, 115, 129, 217, 190, 122, 190, 245),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::unresolved::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xc572_14d2_4c99_a6b3;
  }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnShimArgumentMapping {
  ReturnExpression = 0,
  ReturnExpressionElement = 1,
}

impl ::capnp::introspect::Introspect for ReturnShimArgumentMapping {
  fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Enum(::capnp::introspect::RawEnumSchema { encoded_node: &return_shim_argument_mapping::ENCODED_NODE, annotation_types: return_shim_argument_mapping::get_annotation_types }).into() }
}
impl ::core::convert::From<ReturnShimArgumentMapping> for ::capnp::dynamic_value::Reader<'_> {
  fn from(e: ReturnShimArgumentMapping) -> Self { ::capnp::dynamic_value::Enum::new(e.into(), ::capnp::introspect::RawEnumSchema { encoded_node: &return_shim_argument_mapping::ENCODED_NODE, annotation_types: return_shim_argument_mapping::get_annotation_types }.into()).into() }
}
impl ::core::convert::TryFrom<u16> for ReturnShimArgumentMapping {
  type Error = ::capnp::NotInSchema;
  fn try_from(value: u16) -> ::core::result::Result<Self, <ReturnShimArgumentMapping as ::core::convert::TryFrom<u16>>::Error> {
    match value {
      0 => ::core::result::Result::Ok(Self::ReturnExpression),
      1 => ::core::result::Result::Ok(Self::ReturnExpressionElement),
      n => ::core::result::Result::Err(::capnp::NotInSchema(n)),
    }
  }
}
impl From<ReturnShimArgumentMapping> for u16 {
  #[inline]
  fn from(x: ReturnShimArgumentMapping) -> u16 { x as u16 }
}
impl ::capnp::traits::HasTypeId for ReturnShimArgumentMapping {
  const TYPE_ID: u64 = 0xe85e_ce9d_ffde_78f5u64;
}
mod return_shim_argument_mapping {
pub static ENCODED_NODE: [::capnp::Word; 35] = [
  ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
  ::capnp::word(245, 120, 222, 255, 157, 206, 94, 232),
  ::capnp::word(42, 0, 0, 0, 2, 0, 0, 0),
  ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(21, 0, 0, 0, 34, 2, 0, 0),
  ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(49, 0, 0, 0, 55, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
  ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
  ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
  ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
  ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
  ::capnp::word(112, 58, 82, 101, 116, 117, 114, 110),
  ::capnp::word(83, 104, 105, 109, 65, 114, 103, 117),
  ::capnp::word(109, 101, 110, 116, 77, 97, 112, 112),
  ::capnp::word(105, 110, 103, 0, 0, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
  ::capnp::word(8, 0, 0, 0, 1, 0, 2, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(17, 0, 0, 0, 138, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(17, 0, 0, 0, 194, 0, 0, 0),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(114, 101, 116, 117, 114, 110, 69, 120),
  ::capnp::word(112, 114, 101, 115, 115, 105, 111, 110),
  ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
  ::capnp::word(114, 101, 116, 117, 114, 110, 69, 120),
  ::capnp::word(112, 114, 101, 115, 115, 105, 111, 110),
  ::capnp::word(69, 108, 101, 109, 101, 110, 116, 0),
];
pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
  ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
}
}

pub mod return_shim_callees {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_arguments(self) -> ::capnp::Result<::capnp::enum_list::Reader<'a,crate::pysa_report_capnp::ReturnShimArgumentMapping>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_arguments(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_targets(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_targets(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_call_target::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_targets(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_call_target::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_targets(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_arguments(self) -> ::capnp::Result<::capnp::enum_list::Builder<'a,crate::pysa_report_capnp::ReturnShimArgumentMapping>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_arguments(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::enum_list::Owned<crate::pysa_report_capnp::ReturnShimArgumentMapping>>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_arguments(self, size: u32) -> ::capnp::enum_list::Builder<'a,crate::pysa_report_capnp::ReturnShimArgumentMapping> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_arguments(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 61] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(109, 13, 145, 225, 212, 195, 48, 168),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 82, 101, 116, 117, 114, 110),
      ::capnp::word(83, 104, 105, 109, 67, 97, 108, 108),
      ::capnp::word(101, 101, 115, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(64, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(61, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(60, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(88, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(116, 97, 114, 103, 101, 116, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 41, 63, 90, 218, 66, 21, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 114, 103, 117, 109, 101, 110, 116),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(15, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(245, 120, 222, 255, 157, 206, 94, 232),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_call_target::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::enum_list::Owned<crate::pysa_report_capnp::ReturnShimArgumentMapping> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0xa830_c3d4_e191_0d6d;
  }
}

pub mod expression_callees {
  pub use self::Which::{Call,Identifier,AttributeAccess,Define,FormatStringArtificial,FormatStringStringify,Return};

  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn has_call(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 0 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_identifier(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 1 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_attribute_access(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 2 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_define(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 3 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_format_string_artificial(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 4 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_format_string_stringify(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 5 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn has_return(&self) -> bool {
      if self.reader.get_data_field::<u16>(0) != 6 { return false; }
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichReader<'a,>, ::capnp::NotInSchema> {
      match self.reader.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Call(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Identifier(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(AttributeAccess(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        3 => {
          ::core::result::Result::Ok(Define(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        4 => {
          ::core::result::Result::Ok(FormatStringArtificial(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        5 => {
          ::core::result::Result::Ok(FormatStringStringify(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        6 => {
          ::core::result::Result::Ok(Return(
            ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn set_call(&mut self, value: crate::pysa_report_capnp::call_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_call(self, ) -> crate::pysa_report_capnp::call_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 0);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_call(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 0 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_identifier(&mut self, value: crate::pysa_report_capnp::identifier_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_identifier(self, ) -> crate::pysa_report_capnp::identifier_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 1);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_identifier(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 1 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_attribute_access(&mut self, value: crate::pysa_report_capnp::attribute_access_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 2);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_attribute_access(self, ) -> crate::pysa_report_capnp::attribute_access_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 2);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_attribute_access(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 2 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_define(&mut self, value: crate::pysa_report_capnp::define_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 3);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_define(self, ) -> crate::pysa_report_capnp::define_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 3);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_define(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 3 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_format_string_artificial(&mut self, value: crate::pysa_report_capnp::format_string_artificial_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 4);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_format_string_artificial(self, ) -> crate::pysa_report_capnp::format_string_artificial_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 4);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_format_string_artificial(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 4 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_format_string_stringify(&mut self, value: crate::pysa_report_capnp::format_string_stringify_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 5);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_format_string_stringify(self, ) -> crate::pysa_report_capnp::format_string_stringify_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 5);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_format_string_stringify(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 5 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn set_return(&mut self, value: crate::pysa_report_capnp::return_shim_callees::Reader<'_>) -> ::capnp::Result<()> {
      self.builder.set_data_field::<u16>(0, 6);
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_return(self, ) -> crate::pysa_report_capnp::return_shim_callees::Builder<'a> {
      self.builder.set_data_field::<u16>(0, 6);
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_return(&self) -> bool {
      if self.builder.get_data_field::<u16>(0) != 6 { return false; }
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn which(self) -> ::core::result::Result<WhichBuilder<'a,>, ::capnp::NotInSchema> {
      match self.builder.get_data_field::<u16>(0) {
        0 => {
          ::core::result::Result::Ok(Call(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        1 => {
          ::core::result::Result::Ok(Identifier(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        2 => {
          ::core::result::Result::Ok(AttributeAccess(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        3 => {
          ::core::result::Result::Ok(Define(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        4 => {
          ::core::result::Result::Ok(FormatStringArtificial(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        5 => {
          ::core::result::Result::Ok(FormatStringStringify(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        6 => {
          ::core::result::Result::Ok(Return(
            ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
          ))
        }
        x => ::core::result::Result::Err(::capnp::NotInSchema(x))
      }
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 133] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(223, 140, 189, 152, 223, 125, 32, 156),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 7, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 143, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 69, 120, 112, 114, 101, 115),
      ::capnp::word(115, 105, 111, 110, 67, 97, 108, 108),
      ::capnp::word(101, 101, 115, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(28, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 255, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(181, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(176, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(188, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 254, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(185, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(184, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(196, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 253, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(193, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(192, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(204, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 252, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(201, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(196, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(208, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 251, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(205, 0, 0, 0, 186, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(208, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(220, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 250, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(217, 0, 0, 0, 178, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(220, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(232, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(6, 0, 249, 255, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(229, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(224, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(236, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(99, 97, 108, 108, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(85, 150, 122, 198, 157, 194, 151, 250),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 100, 101, 110, 116, 105, 102, 105),
      ::capnp::word(101, 114, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 95, 40, 104, 0, 220, 240, 165),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 116, 116, 114, 105, 98, 117, 116),
      ::capnp::word(101, 65, 99, 99, 101, 115, 115, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 114, 216, 20, 247, 203, 3, 218),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 101, 102, 105, 110, 101, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(231, 220, 255, 73, 22, 33, 121, 179),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 111, 114, 109, 97, 116, 83, 116),
      ::capnp::word(114, 105, 110, 103, 65, 114, 116, 105),
      ::capnp::word(102, 105, 99, 105, 97, 108, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(67, 149, 110, 38, 253, 142, 93, 160),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 111, 114, 109, 97, 116, 83, 116),
      ::capnp::word(114, 105, 110, 103, 83, 116, 114, 105),
      ::capnp::word(110, 103, 105, 102, 121, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(179, 166, 153, 76, 210, 20, 114, 197),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(114, 101, 116, 117, 114, 110, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 13, 145, 225, 212, 195, 48, 168),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::call_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::identifier_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::attribute_access_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <crate::pysa_report_capnp::define_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <crate::pysa_report_capnp::format_string_artificial_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        5 => <crate::pysa_report_capnp::format_string_stringify_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        6 => <crate::pysa_report_capnp::return_shim_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[0,1,2,3,4,5,6];
    pub static MEMBERS_BY_NAME : &[u16] = &[2,0,3,4,5,1,6];
    pub const TYPE_ID: u64 = 0x9c20_7ddf_98bd_8cdf;
  }
  pub enum Which<A0,A1,A2,A3,A4,A5,A6> {
    Call(A0),
    Identifier(A1),
    AttributeAccess(A2),
    Define(A3),
    FormatStringArtificial(A4),
    FormatStringStringify(A5),
    Return(A6),
  }
  pub type WhichReader<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::call_callees::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::identifier_callees::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::attribute_access_callees::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::define_callees::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::format_string_artificial_callees::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::format_string_stringify_callees::Reader<'a>>,::capnp::Result<crate::pysa_report_capnp::return_shim_callees::Reader<'a>>>;
  pub type WhichBuilder<'a,> = Which<::capnp::Result<crate::pysa_report_capnp::call_callees::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::identifier_callees::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::attribute_access_callees::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::define_callees::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::format_string_artificial_callees::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::format_string_stringify_callees::Builder<'a>>,::capnp::Result<crate::pysa_report_capnp::return_shim_callees::Builder<'a>>>;
}

pub mod call_graph_entry {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_expression_id(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_expression_id(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_callees(self) -> ::capnp::Result<crate::pysa_report_capnp::expression_callees::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_callees(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_expression_id(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_expression_id(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_expression_id(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_expression_id(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_callees(self) -> ::capnp::Result<crate::pysa_report_capnp::expression_callees::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_callees(&mut self, value: crate::pysa_report_capnp::expression_callees::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_callees(self, ) -> crate::pysa_report_capnp::expression_callees::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_callees(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_callees(&self) -> crate::pysa_report_capnp::expression_callees::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 53] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(248, 24, 241, 49, 213, 175, 24, 217),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 202, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 67, 97, 108, 108, 71, 114),
      ::capnp::word(97, 112, 104, 69, 110, 116, 114, 121),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 106, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(101, 120, 112, 114, 101, 115, 115, 105),
      ::capnp::word(111, 110, 73, 100, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 97, 108, 108, 101, 101, 115, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(223, 140, 189, 152, 223, 125, 32, 156),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::expression_callees::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0xd918_afd5_31f1_18f8;
  }
}

pub mod function_call_graph {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_entries(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::call_graph_entry::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_entries(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 2 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_function_id(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_function_id(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_entries(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::call_graph_entry::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_entries(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::call_graph_entry::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_entries(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::call_graph_entry::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_entries(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 57] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(42, 217, 233, 124, 244, 207, 78, 229),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(2, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 67, 97, 108, 108, 71, 114),
      ::capnp::word(97, 112, 104, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(72, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(73, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 110, 116, 114, 105, 101, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(248, 24, 241, 49, 213, 175, 24, 217),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::call_graph_entry::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,0];
    pub const TYPE_ID: u64 = 0xe54e_cff4_7ce9_d92a;
  }
}

pub mod pysa_project_module {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_relative_source_path(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_relative_source_path(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_info_filename(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_info_filename(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_python_version(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_python_version(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
    #[inline]
    pub fn get_platform(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_platform(&self) -> bool {
      !self.reader.get_pointer_field(5).is_null()
    }
    #[inline]
    pub fn get_is_test(self) -> bool {
      self.reader.get_bool_field(32)
    }
    #[inline]
    pub fn get_is_interface(self) -> bool {
      self.reader.get_bool_field(33)
    }
    #[inline]
    pub fn get_is_init(self) -> bool {
      self.reader.get_bool_field(34)
    }
    #[inline]
    pub fn get_is_internal(self) -> bool {
      self.reader.get_bool_field(35)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 6 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_module_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_module_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_source_path(&mut self, value: crate::pysa_report_capnp::source_path::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_source_path(self, ) -> crate::pysa_report_capnp::source_path::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_relative_source_path(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_relative_source_path(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false).unwrap()
    }
    #[inline]
    pub fn init_relative_source_path(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(2).init_text(size)
    }
    #[inline]
    pub fn has_relative_source_path(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_info_filename(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_info_filename(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false).unwrap()
    }
    #[inline]
    pub fn init_info_filename(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(3).init_text(size)
    }
    #[inline]
    pub fn has_info_filename(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_python_version(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_python_version(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false).unwrap()
    }
    #[inline]
    pub fn init_python_version(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(4).init_text(size)
    }
    #[inline]
    pub fn has_python_version(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
    #[inline]
    pub fn get_platform(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_platform(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(5), value, false).unwrap()
    }
    #[inline]
    pub fn init_platform(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(5).init_text(size)
    }
    #[inline]
    pub fn has_platform(&self) -> bool {
      !self.builder.is_pointer_field_null(5)
    }
    #[inline]
    pub fn get_is_test(self) -> bool {
      self.builder.get_bool_field(32)
    }
    #[inline]
    pub fn set_is_test(&mut self, value: bool)  {
      self.builder.set_bool_field(32, value);
    }
    #[inline]
    pub fn get_is_interface(self) -> bool {
      self.builder.get_bool_field(33)
    }
    #[inline]
    pub fn set_is_interface(&mut self, value: bool)  {
      self.builder.set_bool_field(33, value);
    }
    #[inline]
    pub fn get_is_init(self) -> bool {
      self.builder.get_bool_field(34)
    }
    #[inline]
    pub fn set_is_init(&mut self, value: bool)  {
      self.builder.set_bool_field(34, value);
    }
    #[inline]
    pub fn get_is_internal(self) -> bool {
      self.builder.get_bool_field(35)
    }
    #[inline]
    pub fn set_is_internal(&mut self, value: bool)  {
      self.builder.set_bool_field(35, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_source_path(&self) -> crate::pysa_report_capnp::source_path::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 197] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(75, 198, 179, 133, 193, 121, 142, 137),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(6, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 111, 2, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 121, 115, 97, 80, 114),
      ::capnp::word(111, 106, 101, 99, 116, 77, 111, 100),
      ::capnp::word(117, 108, 101, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(37, 1, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(36, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(48, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 1, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(53, 1, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(52, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(64, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(61, 1, 0, 0, 154, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(64, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(76, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(73, 1, 0, 0, 106, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(72, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(84, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(81, 1, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(80, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(92, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(6, 0, 0, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 6, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(89, 1, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(88, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(100, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(7, 0, 0, 0, 32, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(92, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(104, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 33, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 8, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(101, 1, 0, 0, 98, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(112, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(9, 0, 0, 0, 34, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 9, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 1, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(116, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(10, 0, 0, 0, 35, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 10, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(113, 1, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(124, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 78, 97),
      ::capnp::word(109, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(115, 111, 117, 114, 99, 101, 80, 97),
      ::capnp::word(116, 104, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 96, 95, 20, 53, 150, 120, 201),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(114, 101, 108, 97, 116, 105, 118, 101),
      ::capnp::word(83, 111, 117, 114, 99, 101, 80, 97),
      ::capnp::word(116, 104, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 110, 102, 111, 70, 105, 108, 101),
      ::capnp::word(110, 97, 109, 101, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 116, 104, 111, 110, 86, 101),
      ::capnp::word(114, 115, 105, 111, 110, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 108, 97, 116, 102, 111, 114, 109),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 84, 101, 115, 116, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 73, 110, 116, 101, 114, 102),
      ::capnp::word(97, 99, 101, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 73, 110, 105, 116, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 115, 73, 110, 116, 101, 114, 110),
      ::capnp::word(97, 108, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::source_path::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        5 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        6 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        7 => <bool as ::capnp::introspect::Introspect>::introspect(),
        8 => <bool as ::capnp::introspect::Introspect>::introspect(),
        9 => <bool as ::capnp::introspect::Introspect>::introspect(),
        10 => <bool as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5,6,7,8,9,10];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[4,9,8,10,7,0,1,6,5,3,2];
    pub const TYPE_ID: u64 = 0x898e_79c1_85b3_c64b;
  }
}

pub mod project_file {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_modules(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_project_module::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_modules(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_builtin_module_ids(self) -> ::capnp::Result<::capnp::primitive_list::Reader<'a,u32>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_builtin_module_ids(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_object_class_refs(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_object_class_refs(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_dict_class_refs(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_dict_class_refs(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_typing_module_ids(self) -> ::capnp::Result<::capnp::primitive_list::Reader<'a,u32>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_typing_module_ids(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
    #[inline]
    pub fn get_typing_mapping_class_refs(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_typing_mapping_class_refs(&self) -> bool {
      !self.reader.get_pointer_field(5).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 6 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_modules(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_project_module::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_modules(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_project_module::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_modules(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_project_module::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_modules(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_builtin_module_ids(self) -> ::capnp::Result<::capnp::primitive_list::Builder<'a,u32>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_builtin_module_ids(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::primitive_list::Owned<u32>>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_builtin_module_ids(self, size: u32) -> ::capnp::primitive_list::Builder<'a,u32> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_builtin_module_ids(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_object_class_refs(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_object_class_refs(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_object_class_refs(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_object_class_refs(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_dict_class_refs(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_dict_class_refs(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false)
    }
    #[inline]
    pub fn init_dict_class_refs(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(3), size)
    }
    #[inline]
    pub fn has_dict_class_refs(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_typing_module_ids(self) -> ::capnp::Result<::capnp::primitive_list::Builder<'a,u32>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_typing_module_ids(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::primitive_list::Owned<u32>>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false)
    }
    #[inline]
    pub fn init_typing_module_ids(self, size: u32) -> ::capnp::primitive_list::Builder<'a,u32> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(4), size)
    }
    #[inline]
    pub fn has_typing_module_ids(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
    #[inline]
    pub fn get_typing_mapping_class_refs(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(5), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_typing_mapping_class_refs(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_ref::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(5), value, false)
    }
    #[inline]
    pub fn init_typing_mapping_class_refs(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_ref::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(5), size)
    }
    #[inline]
    pub fn has_typing_mapping_class_refs(&self) -> bool {
      !self.builder.is_pointer_field_null(5)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 142] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(174, 156, 108, 204, 119, 227, 96, 251),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(6, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 178, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 87, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 80, 114, 111, 106, 101, 99),
      ::capnp::word(116, 70, 105, 108, 101, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(24, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(153, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(148, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(176, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(173, 0, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(176, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(204, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(201, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(200, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(228, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(225, 0, 0, 0, 114, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(224, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(252, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(249, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(248, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(20, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(17, 1, 0, 0, 186, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(20, 1, 0, 0, 3, 0, 1, 0),
      ::capnp::word(48, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(75, 198, 179, 133, 193, 121, 142, 137),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(98, 117, 105, 108, 116, 105, 110, 77),
      ::capnp::word(111, 100, 117, 108, 101, 73, 100, 115),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(111, 98, 106, 101, 99, 116, 67, 108),
      ::capnp::word(97, 115, 115, 82, 101, 102, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(100, 105, 99, 116, 67, 108, 97, 115),
      ::capnp::word(115, 82, 101, 102, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 121, 112, 105, 110, 103, 77, 111),
      ::capnp::word(100, 117, 108, 101, 73, 100, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 121, 112, 105, 110, 103, 77, 97),
      ::capnp::word(112, 112, 105, 110, 103, 67, 108, 97),
      ::capnp::word(115, 115, 82, 101, 102, 115, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(199, 37, 221, 208, 246, 159, 94, 222),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_project_module::Owned> as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::primitive_list::Owned<u32> as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        4 => <::capnp::primitive_list::Owned<u32> as ::capnp::introspect::Introspect>::introspect(),
        5 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_ref::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[1,3,0,2,5,4];
    pub const TYPE_ID: u64 = 0xfb60_e377_cc6c_9cae;
  }
}

pub mod module_definitions {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_function_definitions(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::function_definition::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_function_definitions(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_class_definitions(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::class_definition::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_class_definitions(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_global_variables(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::global_variable::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_global_variables(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 5 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_module_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_module_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_source_path(&mut self, value: crate::pysa_report_capnp::source_path::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_source_path(self, ) -> crate::pysa_report_capnp::source_path::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_function_definitions(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_definition::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_function_definitions(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::function_definition::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_function_definitions(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_definition::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_function_definitions(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_class_definitions(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_definition::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_class_definitions(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::class_definition::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false)
    }
    #[inline]
    pub fn init_class_definitions(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::class_definition::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(3), size)
    }
    #[inline]
    pub fn has_class_definitions(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_global_variables(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::global_variable::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_global_variables(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::global_variable::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false)
    }
    #[inline]
    pub fn init_global_variables(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::global_variable::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(4), size)
    }
    #[inline]
    pub fn has_global_variables(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_source_path(&self) -> crate::pysa_report_capnp::source_path::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 132] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(136, 91, 71, 204, 133, 52, 146, 199),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(5, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 226, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 87, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 77, 111, 100, 117, 108, 101),
      ::capnp::word(68, 101, 102, 105, 110, 105, 116, 105),
      ::capnp::word(111, 110, 115, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(24, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(153, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(152, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(164, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(161, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(160, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(172, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(169, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(168, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(180, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(177, 0, 0, 0, 162, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(180, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(208, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(205, 0, 0, 0, 138, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(208, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(236, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(5, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 5, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(233, 0, 0, 0, 130, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(232, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(4, 1, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 78, 97),
      ::capnp::word(109, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(115, 111, 117, 114, 99, 101, 80, 97),
      ::capnp::word(116, 104, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 96, 95, 20, 53, 150, 120, 201),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(68, 101, 102, 105, 110, 105, 116, 105),
      ::capnp::word(111, 110, 115, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(17, 100, 67, 113, 216, 204, 43, 172),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 108, 97, 115, 115, 68, 101, 102),
      ::capnp::word(105, 110, 105, 116, 105, 111, 110, 115),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(25, 191, 82, 128, 248, 185, 2, 248),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(103, 108, 111, 98, 97, 108, 86, 97),
      ::capnp::word(114, 105, 97, 98, 108, 101, 115, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(107, 4, 36, 64, 15, 189, 236, 215),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::source_path::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::function_definition::Owned> as ::capnp::introspect::Introspect>::introspect(),
        4 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::class_definition::Owned> as ::capnp::introspect::Introspect>::introspect(),
        5 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::global_variable::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4,5];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[4,3,5,0,1,2];
    pub const TYPE_ID: u64 = 0xc792_3485_cc47_5b88;
  }
}

pub mod location_type_id_entry {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_type_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_location(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_location(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), 0)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_type_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_type_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_location(&self) -> crate::pysa_report_capnp::pysa_location::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(0))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 53] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(7, 11, 69, 222, 140, 144, 151, 236),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 242, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 119, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 76, 111, 99, 97, 116, 105),
      ::capnp::word(111, 110, 84, 121, 112, 101, 73, 100),
      ::capnp::word(69, 110, 116, 114, 121, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(40, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(52, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(44, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(56, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 121, 112, 101, 73, 100, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,1];
    pub const TYPE_ID: u64 = 0xec97_908c_de45_0b07;
  }
}

pub mod function_type_of_expressions {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_types(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::pysa_type::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_types(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_locations(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::location_type_id_entry::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_locations(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_function_id(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_function_id(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_function_id(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_function_id(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_types(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_type::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_types(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::pysa_type::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_types(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::pysa_type::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), size)
    }
    #[inline]
    pub fn has_types(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_locations(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::location_type_id_entry::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_locations(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::location_type_id_entry::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_locations(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::location_type_id_entry::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_locations(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 78] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(233, 138, 87, 196, 135, 199, 84, 228),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 34, 2, 0, 0),
      ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 175, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 70, 117, 110, 99, 116, 105),
      ::capnp::word(111, 110, 84, 121, 112, 101, 79, 102),
      ::capnp::word(69, 120, 112, 114, 101, 115, 115, 105),
      ::capnp::word(111, 110, 115, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(12, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(69, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(68, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(80, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(77, 0, 0, 0, 50, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(72, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(100, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(124, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(73, 100, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(116, 121, 112, 101, 115, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(18, 35, 40, 237, 232, 149, 141, 210),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(7, 11, 69, 222, 140, 144, 151, 236),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::pysa_type::Owned> as ::capnp::introspect::Introspect>::introspect(),
        2 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::location_type_id_entry::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0,2,1];
    pub const TYPE_ID: u64 = 0xe454_c787_c457_8ae9;
  }
}

pub mod module_type_of_expressions {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_functions(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::function_type_of_expressions::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_functions(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_module_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_module_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_source_path(&mut self, value: crate::pysa_report_capnp::source_path::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_source_path(self, ) -> crate::pysa_report_capnp::source_path::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_functions(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_type_of_expressions::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_functions(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::function_type_of_expressions::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_functions(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_type_of_expressions::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_functions(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_source_path(&self) -> crate::pysa_report_capnp::source_path::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 91] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(167, 35, 247, 233, 58, 125, 202, 207),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 18, 2, 0, 0),
      ::capnp::word(53, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(49, 0, 0, 0, 231, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 77, 111, 100, 117, 108, 101),
      ::capnp::word(84, 121, 112, 101, 79, 102, 69, 120),
      ::capnp::word(112, 114, 101, 115, 115, 105, 111, 110),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(116, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(113, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(124, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(121, 0, 0, 0, 82, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(120, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(148, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 78, 97),
      ::capnp::word(109, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(115, 111, 117, 114, 99, 101, 80, 97),
      ::capnp::word(116, 104, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 96, 95, 20, 53, 150, 120, 201),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(102, 117, 110, 99, 116, 105, 111, 110),
      ::capnp::word(115, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(233, 138, 87, 196, 135, 199, 84, 228),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::source_path::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::function_type_of_expressions::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[3,0,1,2];
    pub const TYPE_ID: u64 = 0xcfca_7d3a_e9f7_23a7;
  }
}

pub mod module_call_graphs {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.reader.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_call_graphs(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::function_call_graph::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_call_graphs(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 1, pointers: 3 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_id(self) -> u32 {
      self.builder.get_data_field::<u32>(0)
    }
    #[inline]
    pub fn set_module_id(&mut self, value: u32)  {
      self.builder.set_data_field::<u32>(0, value);
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_module_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_module_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_source_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_source_path(&mut self, value: crate::pysa_report_capnp::source_path::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_source_path(self, ) -> crate::pysa_report_capnp::source_path::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_source_path(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_call_graphs(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_call_graph::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_call_graphs(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::function_call_graph::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_call_graphs(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::function_call_graph::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), size)
    }
    #[inline]
    pub fn has_call_graphs(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_source_path(&self) -> crate::pysa_report_capnp::source_path::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 90] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(198, 246, 102, 55, 151, 81, 18, 193),
      ::capnp::word(42, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(3, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 218, 1, 0, 0),
      ::capnp::word(49, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(45, 0, 0, 0, 231, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 77, 111, 100, 117, 108, 101),
      ::capnp::word(67, 97, 108, 108, 71, 114, 97, 112),
      ::capnp::word(104, 115, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(97, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(108, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(105, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(104, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(116, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(113, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(124, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(121, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(120, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(148, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 73, 100),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 78, 97),
      ::capnp::word(109, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(115, 111, 117, 114, 99, 101, 80, 97),
      ::capnp::word(116, 104, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 96, 95, 20, 53, 150, 120, 201),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(99, 97, 108, 108, 71, 114, 97, 112),
      ::capnp::word(104, 115, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(42, 217, 233, 124, 244, 207, 78, 229),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <u32 as ::capnp::introspect::Introspect>::introspect(),
        1 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::source_path::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::function_call_graph::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[3,0,1,2];
    pub const TYPE_ID: u64 = 0xc112_5197_3766_f6c6;
  }
}

pub mod type_error {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
    #[inline]
    pub fn get_module_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_module_path(&self) -> bool {
      !self.reader.get_pointer_field(1).is_null()
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.reader.get_pointer_field(2).is_null()
    }
    #[inline]
    pub fn get_kind(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_kind(&self) -> bool {
      !self.reader.get_pointer_field(3).is_null()
    }
    #[inline]
    pub fn get_message(self) -> ::capnp::Result<::capnp::text::Reader<'a>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_message(&self) -> bool {
      !self.reader.get_pointer_field(4).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 5 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_module_name(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_module_name(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false).unwrap()
    }
    #[inline]
    pub fn init_module_name(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(0).init_text(size)
    }
    #[inline]
    pub fn has_module_name(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
    #[inline]
    pub fn get_module_path(self) -> ::capnp::Result<crate::pysa_report_capnp::source_path::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(1), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_module_path(&mut self, value: crate::pysa_report_capnp::source_path::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(1), value, false)
    }
    #[inline]
    pub fn init_module_path(self, ) -> crate::pysa_report_capnp::source_path::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(1), 0)
    }
    #[inline]
    pub fn has_module_path(&self) -> bool {
      !self.builder.is_pointer_field_null(1)
    }
    #[inline]
    pub fn get_location(self) -> ::capnp::Result<crate::pysa_report_capnp::pysa_location::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(2), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_location(&mut self, value: crate::pysa_report_capnp::pysa_location::Reader<'_>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(2), value, false)
    }
    #[inline]
    pub fn init_location(self, ) -> crate::pysa_report_capnp::pysa_location::Builder<'a> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(2), 0)
    }
    #[inline]
    pub fn has_location(&self) -> bool {
      !self.builder.is_pointer_field_null(2)
    }
    #[inline]
    pub fn get_kind(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(3), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_kind(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(3), value, false).unwrap()
    }
    #[inline]
    pub fn init_kind(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(3).init_text(size)
    }
    #[inline]
    pub fn has_kind(&self) -> bool {
      !self.builder.is_pointer_field_null(3)
    }
    #[inline]
    pub fn get_message(self) -> ::capnp::Result<::capnp::text::Builder<'a>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(4), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_message(&mut self, value: impl ::capnp::traits::SetterInput<::capnp::text::Owned>)  {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(4), value, false).unwrap()
    }
    #[inline]
    pub fn init_message(self, size: u32) -> ::capnp::text::Builder<'a> {
      self.builder.get_pointer_field(4).init_text(size)
    }
    #[inline]
    pub fn has_message(&self) -> bool {
      !self.builder.is_pointer_field_null(4)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
    pub fn get_module_path(&self) -> crate::pysa_report_capnp::source_path::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(1))
    }
    pub fn get_location(&self) -> crate::pysa_report_capnp::pysa_location::Pipeline {
      ::capnp::capability::FromTypelessPipeline::new(self._typeless.get_pointer_field(2))
    }
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 99] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(183, 214, 135, 70, 127, 62, 131, 207),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(5, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 162, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 31, 1, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 84, 121, 112, 101, 69, 114),
      ::capnp::word(114, 111, 114, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(20, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(124, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(136, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(1, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 1, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(133, 0, 0, 0, 90, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(132, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(144, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(2, 0, 0, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 2, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(141, 0, 0, 0, 74, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(140, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(152, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(3, 0, 0, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 3, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(149, 0, 0, 0, 42, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(144, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(156, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 4, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(153, 0, 0, 0, 66, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(148, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(160, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 78, 97),
      ::capnp::word(109, 101, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 111, 100, 117, 108, 101, 80, 97),
      ::capnp::word(116, 104, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(125, 96, 95, 20, 53, 150, 120, 201),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(108, 111, 99, 97, 116, 105, 111, 110),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(96, 138, 229, 243, 105, 201, 9, 195),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(107, 105, 110, 100, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(109, 101, 115, 115, 97, 103, 101, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(12, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        1 => <crate::pysa_report_capnp::source_path::Owned as ::capnp::introspect::Introspect>::introspect(),
        2 => <crate::pysa_report_capnp::pysa_location::Owned as ::capnp::introspect::Introspect>::introspect(),
        3 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        4 => <::capnp::text::Owned as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0,1,2,3,4];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[3,2,4,0,1];
    pub const TYPE_ID: u64 = 0xcf83_3e7f_4687_d6b7;
  }
}

pub mod type_errors {
  #[derive(Copy, Clone)]
  pub struct Owned(());
  impl ::capnp::introspect::Introspect for Owned { fn introspect() -> ::capnp::introspect::Type { ::capnp::introspect::TypeVariant::Struct(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types, annotation_types: _private::get_annotation_types }).into() } }
  impl ::capnp::traits::Owned for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::OwnedStruct for Owned { type Reader<'a> = Reader<'a>; type Builder<'a> = Builder<'a>; }
  impl ::capnp::traits::Pipelined for Owned { type Pipeline = Pipeline; }

  pub struct Reader<'a> { reader: ::capnp::private::layout::StructReader<'a> }
  impl <> ::core::marker::Copy for Reader<'_,>  {}
  impl <> ::core::clone::Clone for Reader<'_,>  {
    fn clone(&self) -> Self { *self }
  }

  impl <> ::capnp::traits::HasTypeId for Reader<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructReader<'a>> for Reader<'a,>  {
    fn from(reader: ::capnp::private::layout::StructReader<'a>) -> Self {
      Self { reader,  }
    }
  }

  impl <'a,> ::core::convert::From<Reader<'a,>> for ::capnp::dynamic_value::Reader<'a>  {
    fn from(reader: Reader<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Reader::new(reader.reader, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <> ::core::fmt::Debug for Reader<'_,>  {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::result::Result<(), ::core::fmt::Error> {
      core::fmt::Debug::fmt(&::core::convert::Into::<::capnp::dynamic_value::Reader<'_>>::into(*self), f)
    }
  }

  impl <'a,> ::capnp::traits::FromPointerReader<'a> for Reader<'a,>  {
    fn get_from_pointer(reader: &::capnp::private::layout::PointerReader<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(reader.get_struct(default)?.into())
    }
  }

  impl <'a,> ::capnp::traits::IntoInternalStructReader<'a> for Reader<'a,>  {
    fn into_internal_struct_reader(self) -> ::capnp::private::layout::StructReader<'a> {
      self.reader
    }
  }

  impl <'a,> ::capnp::traits::Imbue<'a> for Reader<'a,>  {
    fn imbue(&mut self, cap_table: &'a ::capnp::private::layout::CapTable) {
      self.reader.imbue(::capnp::private::layout::CapTableReader::Plain(cap_table))
    }
  }

  impl <'a,> Reader<'a,>  {
    pub fn reborrow(&self) -> Reader<'_,> {
      Self { .. *self }
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.reader.total_size()
    }
    #[inline]
    pub fn get_errors(self) -> ::capnp::Result<::capnp::struct_list::Reader<'a,crate::pysa_report_capnp::type_error::Owned>> {
      ::capnp::traits::FromPointerReader::get_from_pointer(&self.reader.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn has_errors(&self) -> bool {
      !self.reader.get_pointer_field(0).is_null()
    }
  }

  pub struct Builder<'a> { builder: ::capnp::private::layout::StructBuilder<'a> }
  impl <> ::capnp::traits::HasStructSize for Builder<'_,>  {
    const STRUCT_SIZE: ::capnp::private::layout::StructSize = ::capnp::private::layout::StructSize { data: 0, pointers: 1 };
  }
  impl <> ::capnp::traits::HasTypeId for Builder<'_,>  {
    const TYPE_ID: u64 = _private::TYPE_ID;
  }
  impl <'a,> ::core::convert::From<::capnp::private::layout::StructBuilder<'a>> for Builder<'a,>  {
    fn from(builder: ::capnp::private::layout::StructBuilder<'a>) -> Self {
      Self { builder,  }
    }
  }

  impl <'a,> ::core::convert::From<Builder<'a,>> for ::capnp::dynamic_value::Builder<'a>  {
    fn from(builder: Builder<'a,>) -> Self {
      Self::Struct(::capnp::dynamic_struct::Builder::new(builder.builder, ::capnp::schema::StructSchema::new(::capnp::introspect::RawBrandedStructSchema { generic: &_private::RAW_SCHEMA, field_types: _private::get_field_types::<>, annotation_types: _private::get_annotation_types::<>})))
    }
  }

  impl <'a,> ::capnp::traits::ImbueMut<'a> for Builder<'a,>  {
    fn imbue_mut(&mut self, cap_table: &'a mut ::capnp::private::layout::CapTable) {
      self.builder.imbue(::capnp::private::layout::CapTableBuilder::Plain(cap_table))
    }
  }

  impl <'a,> ::capnp::traits::FromPointerBuilder<'a> for Builder<'a,>  {
    fn init_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, _size: u32) -> Self {
      builder.init_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE).into()
    }
    fn get_from_pointer(builder: ::capnp::private::layout::PointerBuilder<'a>, default: ::core::option::Option<&'a [::capnp::Word]>) -> ::capnp::Result<Self> {
      ::core::result::Result::Ok(builder.get_struct(<Self as ::capnp::traits::HasStructSize>::STRUCT_SIZE, default)?.into())
    }
  }

  impl <> ::capnp::traits::SetterInput<Owned<>> for Reader<'_,>  {
    fn set_pointer_builder(mut pointer: ::capnp::private::layout::PointerBuilder<'_>, value: Self, canonicalize: bool) -> ::capnp::Result<()> { pointer.set_struct(&value.reader, canonicalize) }
  }

  impl <'a,> Builder<'a,>  {
    pub fn into_reader(self) -> Reader<'a,> {
      self.builder.into_reader().into()
    }
    pub fn reborrow(&mut self) -> Builder<'_,> {
      Builder { builder: self.builder.reborrow() }
    }
    pub fn reborrow_as_reader(&self) -> Reader<'_,> {
      self.builder.as_reader().into()
    }

    pub fn total_size(&self) -> ::capnp::Result<::capnp::MessageSize> {
      self.builder.as_reader().total_size()
    }
    #[inline]
    pub fn get_errors(self) -> ::capnp::Result<::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::type_error::Owned>> {
      ::capnp::traits::FromPointerBuilder::get_from_pointer(self.builder.get_pointer_field(0), ::core::option::Option::None)
    }
    #[inline]
    pub fn set_errors(&mut self, value: ::capnp::struct_list::Reader<'_,crate::pysa_report_capnp::type_error::Owned>) -> ::capnp::Result<()> {
      ::capnp::traits::SetterInput::set_pointer_builder(self.builder.reborrow().get_pointer_field(0), value, false)
    }
    #[inline]
    pub fn init_errors(self, size: u32) -> ::capnp::struct_list::Builder<'a,crate::pysa_report_capnp::type_error::Owned> {
      ::capnp::traits::FromPointerBuilder::init_pointer(self.builder.get_pointer_field(0), size)
    }
    #[inline]
    pub fn has_errors(&self) -> bool {
      !self.builder.is_pointer_field_null(0)
    }
  }

  pub struct Pipeline { _typeless: ::capnp::any_pointer::Pipeline }
  impl ::capnp::capability::FromTypelessPipeline for Pipeline {
    fn new(typeless: ::capnp::any_pointer::Pipeline) -> Self {
      Self { _typeless: typeless,  }
    }
  }
  impl Pipeline  {
  }
  mod _private {
    pub static ENCODED_NODE: [::capnp::Word; 40] = [
      ::capnp::word(0, 0, 0, 0, 5, 0, 6, 0),
      ::capnp::word(249, 69, 12, 39, 145, 214, 224, 194),
      ::capnp::word(42, 0, 0, 0, 1, 0, 0, 0),
      ::capnp::word(54, 194, 173, 46, 67, 240, 114, 129),
      ::capnp::word(1, 0, 7, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(21, 0, 0, 0, 170, 1, 0, 0),
      ::capnp::word(45, 0, 0, 0, 7, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(41, 0, 0, 0, 63, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(112, 121, 114, 101, 102, 108, 121, 47),
      ::capnp::word(108, 105, 98, 47, 114, 101, 112, 111),
      ::capnp::word(114, 116, 47, 112, 121, 115, 97, 47),
      ::capnp::word(112, 121, 115, 97, 95, 114, 101, 112),
      ::capnp::word(111, 114, 116, 46, 99, 97, 112, 110),
      ::capnp::word(112, 58, 84, 121, 112, 101, 69, 114),
      ::capnp::word(114, 111, 114, 115, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 1, 0, 1, 0),
      ::capnp::word(4, 0, 0, 0, 3, 0, 4, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 1, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(13, 0, 0, 0, 58, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(8, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(36, 0, 0, 0, 2, 0, 1, 0),
      ::capnp::word(101, 114, 114, 111, 114, 115, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 3, 0, 1, 0),
      ::capnp::word(16, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(183, 214, 135, 70, 127, 62, 131, 207),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(14, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
      ::capnp::word(0, 0, 0, 0, 0, 0, 0, 0),
    ];
    pub fn get_field_types(index: u16) -> ::capnp::introspect::Type {
      match index {
        0 => <::capnp::struct_list::Owned<crate::pysa_report_capnp::type_error::Owned> as ::capnp::introspect::Introspect>::introspect(),
        _ => ::capnp::introspect::panic_invalid_field_index(index),
      }
    }
    pub fn get_annotation_types(child_index: Option<u16>, index: u32) -> ::capnp::introspect::Type {
      ::capnp::introspect::panic_invalid_annotation_indices(child_index, index)
    }
    pub static ARENA: ::capnp::private::arena::GeneratedCodeArena = ::capnp::private::arena::GeneratedCodeArena::new(&ENCODED_NODE);
    pub static RAW_SCHEMA: ::capnp::introspect::RawStructSchema = ::capnp::introspect::RawStructSchema::new(
      &ARENA,
      NONUNION_MEMBERS,
      MEMBERS_BY_DISCRIMINANT,
      MEMBERS_BY_NAME
    );
    pub static NONUNION_MEMBERS : &[u16] = &[0];
    pub static MEMBERS_BY_DISCRIMINANT : &[u16] = &[];
    pub static MEMBERS_BY_NAME : &[u16] = &[0];
    pub const TYPE_ID: u64 = 0xc2e0_d691_270c_45f9;
  }
}
