# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

@0x8172f0432eadc236;

# Cap'n Proto schema for pyrefly's Pysa report output.
#
# After modifying this schema, regenerate pysa_report_capnp.rs by running:
#   ./facebook/generate_pysa_report_capnp.sh

struct SourcePath {
  union {
    fileSystem                @0 :Text;
    namespace                 @1 :Text;
    memory                    @2 :Text;
    bundledTypeshed           @3 :Text;
    bundledTypeshedThirdParty @4 :Text;
    bundledThirdParty         @5 :Text;
  }
}

struct PysaLocation {
  line    @0 :UInt32;
  col     @1 :UInt32;
  endLine @2 :UInt32;
  endCol  @3 :UInt32;
}

struct ClassRef {
  moduleId @0 :UInt32;
  classId  @1 :UInt32;
}

struct FunctionRef {
  moduleId   @0 :UInt32;
  functionId @1 :Text;  # FunctionId serialized via serialize_to_string()
}

struct GlobalVariableRef {
  moduleId @0 :UInt32;
  name     @1 :Text;
}

enum TypeModifier {
  optional              @0;
  coroutine             @1;
  awaitable             @2;
  typeVariableBound     @3;
  typeVariableConstraint @4;
  type                  @5;
}

struct ClassWithModifiers {
  class     @0 :ClassRef;
  modifiers @1 :List(TypeModifier);
}

struct ClassNamesFromType {
  classes      @0 :List(ClassWithModifiers);
  isExhaustive @1 :Bool;
}

struct ScalarTypeProperties {
  isBool  @0 :Bool;
  isInt   @1 :Bool;
  isFloat @2 :Bool;
  isEnum  @3 :Bool;
}

struct PysaType {
  string               @0 :Text;
  scalarTypeProperties @1 :ScalarTypeProperties;
  classNames           @2 :ClassNamesFromType;
}

struct ScopeParent {
  union {
    function @0 :PysaLocation;
    class    @1 :PysaLocation;
    topLevel @2 :Void;
  }
}

struct FunctionParameter {
  union {
    posOnly @0 :PosOnlyParam;
    pos     @1 :PosParam;
    varArg  @2 :VarArgParam;
    kwOnly  @3 :KwOnlyParam;
    kwargs  @4 :KwargsParam;
  }

  struct PosOnlyParam {
    name       @0 :Text;       # null if absent
    annotation @1 :PysaType;
    required   @2 :Bool;
  }

  struct PosParam {
    name       @0 :Text;
    annotation @1 :PysaType;
    required   @2 :Bool;
  }

  struct VarArgParam {
    name       @0 :Text;       # null if absent
    annotation @1 :PysaType;
  }

  struct KwOnlyParam {
    name       @0 :Text;
    annotation @1 :PysaType;
    required   @2 :Bool;
  }

  struct KwargsParam {
    name       @0 :Text;       # null if absent
    annotation @1 :PysaType;
  }
}

struct FunctionParameters {
  union {
    list      @0 :List(FunctionParameter);
    ellipsis  @1 :Void;
    paramSpec @2 :Void;
  }
}

struct FunctionSignature {
  parameters       @0 :FunctionParameters;
  returnAnnotation @1 :PysaType;
}

struct FunctionBaseDefinition {
  name              @0 :Text;
  parent            @1 :ScopeParent;
  isOverload        @2 :Bool;
  isStaticmethod    @3 :Bool;
  isClassmethod     @4 :Bool;
  isPropertyGetter  @5 :Bool;
  isPropertySetter  @6 :Bool;
  isStub            @7 :Bool;
  isDefStatement    @8 :Bool;
  definingClass     @9 :ClassRef;  # null if absent
}

struct CapturedVariableRef {
  outerFunction @0 :FunctionRef;
  name          @1 :Text;
}

struct Target {
  union {
    function     @0 :FunctionRef;
    overrides    @1 :FunctionRef;
    formatString @2 :Void;
  }
}

enum ImplicitReceiver {
  trueWithClassReceiver  @0;
  trueWithObjectReceiver @1;
  false                  @2;
}

struct PysaCallTarget {
  target             @0 :Target;
  implicitReceiver   @1 :ImplicitReceiver;
  implicitDunderCall @2 :Bool;
  receiverClass      @3 :ClassRef;  # null if absent
  isClassMethod      @4 :Bool;
  isStaticMethod     @5 :Bool;
  returnType         @6 :ScalarTypeProperties;
}

struct DecoratorCallee {
  location @0 :PysaLocation;
  targets  @1 :List(Target);
}

struct FunctionDefinition {
  # FunctionBaseDefinition fields (flattened)
  name              @0  :Text;
  parent            @1  :ScopeParent;
  isOverload        @2  :Bool;
  isStaticmethod    @3  :Bool;
  isClassmethod     @4  :Bool;
  isPropertyGetter  @5  :Bool;
  isPropertySetter  @6  :Bool;
  isStub            @7  :Bool;
  isDefStatement    @8  :Bool;
  definingClass     @9  :ClassRef;  # null if absent

  # FunctionDefinition-specific fields
  functionId              @10 :Text;  # FunctionId serialized as string
  undecoratedSignatures   @11 :List(FunctionSignature);
  capturedVariables       @12 :List(CapturedVariableRef);
  decoratorCallees        @13 :List(DecoratorCallee);
  overriddenBaseMethod    @14 :FunctionRef;  # null if absent
}

enum PysaClassFieldDeclaration {
  none                      @0;
  declaredByAnnotation      @1;
  declaredWithoutAnnotation @2;
  assignedInBody            @3;
  definedWithoutAssign      @4;
  definedInMethod           @5;
}

struct PysaClassField {
  name              @0 :Text;  # inlined from HashMap key
  type             @1 :PysaType;
  explicitAnnotation @2 :Text;  # null if absent
  location          @3 :PysaLocation;  # null if absent
  declarationKind   @4 :PysaClassFieldDeclaration;
}

struct PysaClassMro {
  union {
    resolved @0 :List(ClassRef);
    cyclic   @1 :Void;
  }
}

struct ClassDefinition {
  location         @0  :PysaLocation;  # inlined from HashMap key
  classId          @1  :UInt32;
  name             @2  :Text;
  bases            @3  :List(ClassRef);
  mro              @4  :PysaClassMro;
  parent           @5  :ScopeParent;
  isSynthesized    @6  :Bool;
  isDataclass      @7  :Bool;
  isNamedTuple     @8  :Bool;
  isTypedDict      @9  :Bool;
  fields           @10 :List(PysaClassField);
  decoratorCallees @11 :List(DecoratorCallee);
}

struct GlobalVariable {
  name     @0 :Text;           # inlined from HashMap key
  type     @1 :PysaType;       # null if absent
  location @2 :PysaLocation;
}

enum UnresolvedReason {
  lambdaArgument                          @0;
  unexpectedPyreflyTarget                 @1;
  emptyPyreflyCallTarget                  @2;
  unknownClassField                       @3;
  classFieldOnlyExistInObject             @4;
  unsupportedFunctionTarget               @5;
  unexpectedDefiningClass                 @6;
  unexpectedInitMethod                    @7;
  unexpectedNewMethod                     @8;
  unexpectedCalleeExpression              @9;
  unresolvedMagicDunderAttr               @10;
  unresolvedMagicDunderAttrDueToNoBase    @11;
  unresolvedMagicDunderAttrDueToNoAttribute @12;
  mixed                                   @13;
}

struct Unresolved {
  union {
    false @0 :Void;
    true  @1 :UnresolvedReason;
  }
}

struct HigherOrderParameter {
  index       @0 :UInt32;
  callTargets @1 :List(PysaCallTarget);
  unresolved  @2 :Unresolved;
}

struct CallCallees {
  callTargets          @0 :List(PysaCallTarget);
  initTargets          @1 :List(PysaCallTarget);
  newTargets           @2 :List(PysaCallTarget);
  higherOrderParameters @3 :List(HigherOrderParameter);
  unresolved           @4 :Unresolved;
}

struct AttributeAccessCallees {
  ifCalled        @0 :CallCallees;
  propertySetters @1 :List(PysaCallTarget);
  propertyGetters @2 :List(PysaCallTarget);
  globalTargets   @3 :List(GlobalVariableRef);
  isAttribute     @4 :Bool;
}

struct IdentifierCallees {
  ifCalled          @0 :CallCallees;
  globalTargets     @1 :List(GlobalVariableRef);
  capturedVariables @2 :List(CapturedVariableRef);
}

struct DefineCallees {
  defineTargets @0 :List(PysaCallTarget);
}

struct FormatStringArtificialCallees {
  targets @0 :List(PysaCallTarget);
}

struct FormatStringStringifyCallees {
  targets    @0 :List(PysaCallTarget);
  unresolved @1 :Unresolved;
}

enum ReturnShimArgumentMapping {
  returnExpression        @0;
  returnExpressionElement @1;
}

struct ReturnShimCallees {
  targets   @0 :List(PysaCallTarget);
  arguments @1 :List(ReturnShimArgumentMapping);
}

struct ExpressionCallees {
  union {
    call                    @0 :CallCallees;
    identifier              @1 :IdentifierCallees;
    attributeAccess         @2 :AttributeAccessCallees;
    define                  @3 :DefineCallees;
    formatStringArtificial  @4 :FormatStringArtificialCallees;
    formatStringStringify   @5 :FormatStringStringifyCallees;
    return                  @6 :ReturnShimCallees;
  }
}

struct CallGraphEntry {
  expressionId @0 :Text;  # ExpressionIdentifier serialized as string
  callees      @1 :ExpressionCallees;
}

struct FunctionCallGraph {
  functionId @0 :Text;  # FunctionId serialized as string
  entries    @1 :List(CallGraphEntry);
}

struct PysaProjectModule {
  moduleId           @0  :UInt32;
  moduleName         @1  :Text;
  sourcePath         @2  :SourcePath;
  relativeSourcePath @3  :Text;   # null if absent
  infoFilename       @4  :Text;   # null if absent
  pythonVersion      @5  :Text;
  platform           @6  :Text;
  isTest             @7  :Bool;
  isInterface        @8  :Bool;
  isInit             @9  :Bool;
  isInternal         @10 :Bool;
}

struct ProjectFile {
  modules                @0 :List(PysaProjectModule);
  builtinModuleIds       @1 :List(UInt32);
  objectClassRefs        @2 :List(ClassRef);
  dictClassRefs          @3 :List(ClassRef);
  typingModuleIds        @4 :List(UInt32);
  typingMappingClassRefs @5 :List(ClassRef);
}

struct ModuleDefinitions {
  moduleId            @0 :UInt32;
  moduleName          @1 :Text;
  sourcePath          @2 :SourcePath;
  functionDefinitions @3 :List(FunctionDefinition);
  classDefinitions    @4 :List(ClassDefinition);
  globalVariables     @5 :List(GlobalVariable);
}

struct TypeOfExpressionEntry {
  location @0 :PysaLocation;
  type     @1 :PysaType;
}

struct ModuleTypeOfExpressions {
  moduleId         @0 :UInt32;
  moduleName       @1 :Text;
  sourcePath       @2 :SourcePath;
  typeOfExpression  @3 :List(TypeOfExpressionEntry);
}

struct ModuleCallGraphs {
  moduleId      @0 :UInt32;
  moduleName    @1 :Text;
  sourcePath    @2 :SourcePath;
  callGraphs    @3 :List(FunctionCallGraph);
}

struct TypeError {
  moduleName @0 :Text;
  modulePath @1 :SourcePath;
  location   @2 :PysaLocation;
  kind       @3 :Text;   # ErrorKind serialized as string
  message    @4 :Text;
}

struct TypeErrors {
  errors        @0 :List(TypeError);
}
