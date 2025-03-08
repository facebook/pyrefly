"use strict";(self.webpackChunknew_website=self.webpackChunknew_website||[]).push([[3757],{3757:(e,t,n)=>{n.r(t),n.d(t,{default:()=>O});var r=n(96540),l=(n(86025),n(82776));n(32992);const o="editorContainer_KqyR",a="toolbar_hH4G",s="tryEditor_uaey",i="code_d3tC",u="results_FNre",c="resultBody_V266",m="tabs_u867",d="tab_a7pb",g="selectedTab_xnhj",p="loader_Z7Ct",f="bounce1_NjUp",h="bounce2_SM41",y="errors_bNYi",v="msgType_a5Tw";n(8785);var E=n(20053);function N(e){let{error:t}=e;const{startLineNumber:n,startColumn:l,endLineNumber:o,endColumn:a}=t;let s;s=n===o?l===a?n+":"+l:n+":"+l+"-"+a:n+":"+l+"-"+o+":"+a;const i=s+": "+t.message;return r.createElement("span",{className:v},i)}function b(e){let{loading:t,errors:n,internalError:l}=e;const[o,s]=(0,r.useState)("errors");return r.createElement("div",{className:u},r.createElement("div",{className:a},r.createElement("ul",{className:m},r.createElement("li",{className:(0,E.default)(d,"errors"===o&&g),onClick:()=>s("errors")},"Errors"),r.createElement("li",{className:(0,E.default)(d,"json"===o&&g),onClick:()=>s("json")},"JSON"))),t&&r.createElement("div",null,r.createElement("div",{className:p},r.createElement("div",{className:f}),r.createElement("div",{className:h}),r.createElement("div",null))),!t&&"errors"===o&&r.createElement("pre",{className:(0,E.default)(c,y)},r.createElement("ul",null,l?r.createElement("li",null,"Pyrefly encountered an internal error: ",l,"."):null==n?r.createElement("li",null,"Pyrefly failed to fetch errors."):0===(null==n?void 0:n.length)?r.createElement("li",null,"No errors!"):n.map(((e,t)=>r.createElement("li",{key:t},r.createElement(N,{key:t,error:e})))))),!t&&"json"===o&&r.createElement("pre",{className:c},JSON.stringify(n,null,2)))}var w=n(84138);const C=JSON.parse('{"@generated":"copied from fbsource/vscode/vscode-extensions/flow/syntaxes","comments":{"lineComment":"//","blockComment":["/*","*/"]},"brackets":[["{","}"],["[","]"],["(",")"]],"autoClosingPairs":[{"COMMENT":"The {} rule will add the }, only need to add the | here.","open":"{|","close":"|"},{"open":"{","close":"}"},{"open":"[","close":"]"},{"open":"(","close":")"},{"open":"\'","close":"\'","notIn":["string","comment"]},{"open":"\\"","close":"\\"","notIn":["string"]},{"open":"`","close":"`","notIn":["string","comment"]},{"open":"/**","close":" */","notIn":["string"]}],"surroundingPairs":[["{","}"],["[","]"],["(",")"],["\'","\'"],["\\"","\\""],["`","`"]],"autoCloseBefore":";:.,=}])>` \\n\\t","folding":{"markers":{"start":"^\\\\s*//\\\\s*#?region\\\\b","end":"^\\\\s*//\\\\s*#?endregion\\\\b"}}}'),S=(e,t)=>{throw"not implemented"},_=new Map;const M=(e,t)=>null,P=new Map;let k=(e,t)=>null;const I=new Map;const x=()=>[],H=new Map;w.languages.register({id:"python",extensions:[".py"],aliases:["Python"]}),w.languages.setLanguageConfiguration("python",C);w.languages.getEncodedLanguageId("python");w.languages.registerCompletionItemProvider("python",{triggerCharacters:[".","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6","7","8","9","[",'"',"'"],provideCompletionItems(e,t){try{var n;const r=(null!=(n=_.get(e))?n:S)(t.lineNumber,t.column);return console.log("completion",t,r),{suggestions:r.map((e=>({...e,insertText:e.label})))}}catch(r){return console.error(r),null}}}),w.languages.registerDefinitionProvider("python",{provideDefinition(e,t){try{var n;const r=(null!=(n=P.get(e))?n:M)(t.lineNumber,t.column);return null!=r?{uri:e.uri,range:r}:null}catch(r){return console.error(r),null}}}),w.languages.registerHoverProvider("python",{provideHover(e,t){var n;return(null!=(n=I.get(e))?n:k)(t.lineNumber,t.column)}}),w.languages.registerInlayHintsProvider("python",{provideInlayHints(e){var t;return{hints:(null!=(t=H.get(e))?t:x)()}}}),l.wG.config({monaco:w});n(78478);const L='\nfrom typing import *\n\ndef test(x: int):\n  return f"{x}"\n\n# reveal_type will produce a type error that tells you the type Pyre has\n# computed for the argument (in this case, int)\nreveal_type(test(42))\n'.trimStart(),T=("undefined"!=typeof window?n.e(8355).then(n.bind(n,18355)):new Promise((e=>{}))).then((async e=>(await e.default(),e))).catch((e=>console.log(e)));function O(e){let{sampleFilename:t,editorHeight:n="auto",codeSample:a=L,showErrorPanel:u=!0}=e;const c=(0,r.useRef)(null),[m,d]=(0,r.useState)([]),[g,p]=(0,r.useState)(""),[f,h]=(0,r.useState)(!0),[y,v]=(0,r.useState)(null),[E,N]=(0,r.useState)(n),[C,S]=(0,r.useState)(null);function M(){const e=w.editor.getModels().filter((e=>{var n;return(null==e||null==(n=e.uri)?void 0:n.path)==="/"+t}))[0];return null!=e&&e.setValue(e.getValue()),e}function k(){if(null==C||null==y)return;C.getValue();!function(e,t){_.set(e,t)}(C,((e,t)=>y.autoComplete(e,t))),function(e,t){P.set(e,t)}(C,((e,t)=>y.gotoDefinition(e,t))),function(e,t){I.set(e,t)}(C,((e,t)=>y.queryType(e,t))),function(e,t){H.set(e,t)}(C,(()=>y.inlayHint()));try{y.updateSource(C.getValue());const e=y.getErrors();w.editor.setModelMarkers(C,"default",y.getErrors()),p(""),d(e)}catch(e){console.error(e),p(JSON.stringify(e)),d([])}}return(0,r.useEffect)((()=>{h(!0),T.then((e=>{v(new e.State),h(!1),p("")})).catch((e=>{h(!1),p(JSON.stringify(e))}))}),[]),w.editor.onDidCreateModel((e=>{const t=M();S(t),k()})),(0,r.useEffect)((()=>{k()}),[y,C]),r.createElement("div",{className:s},r.createElement("div",{className:i},r.createElement("div",{className:o},r.createElement(l.Ay,{defaultPath:t,defaultValue:a,defaultLanguage:"python",theme:"vs-light",height:E,onChange:k,onMount:function(e){const t=M();S(t),"auto"===n&&N(Math.max(50,e.getContentHeight())),c.current=e},options:{minimap:{enabled:!1},hover:{enabled:!0,above:!1},scrollBeyondLastLine:!1,overviewRulerBorder:!1,scrollbar:{alwaysConsumeMouseWheel:!1}}}))),u&&r.createElement(b,{loading:f,errors:m,internalError:g}))}}}]);