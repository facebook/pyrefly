"use strict";(self.webpackChunknew_website=self.webpackChunknew_website||[]).push([[10,1774],{50010:(e,t,a)=>{a.r(t),a.d(t,{default:()=>Se});var n=a(96540),l=a(20053),r=a(69024),o=a(17559),i=a(2967),c=a(84142),s=a(32252),d=a(26588),m=a(68315),u=a(21312),b=a(23104),p=a(75062);const h="backToTopButton_sjWU",f="backToTopButtonShow_xfvO";function E(){const{shown:e,scrollToTop:t}=function(e){let{threshold:t}=e;const[a,l]=(0,n.useState)(!1),r=(0,n.useRef)(!1),{startScroll:o,cancelScroll:i}=(0,b.gk)();return(0,b.Mq)(((e,a)=>{let{scrollY:n}=e;const o=null==a?void 0:a.scrollY;o&&(r.current?r.current=!1:n>=o?(i(),l(!1)):n<t?l(!1):n+window.innerHeight<document.documentElement.scrollHeight&&l(!0))})),(0,p.$)((e=>{e.location.hash&&(r.current=!0,l(!1))})),{shown:a,scrollToTop:()=>o(0)}}({threshold:300});return n.createElement("button",{"aria-label":(0,u.translate)({id:"theme.BackToTopButton.buttonAriaLabel",message:"Scroll back to top",description:"The ARIA label for the back to top button"}),className:(0,l.default)("clean-btn",o.G.common.backToTopButton,h,e&&f),type:"button",onClick:t})}var v=a(53109),g=a(72681),_=a(24581),C=a(6342),k=a(23465),S=a(58168);function N(e){return n.createElement("svg",(0,S.A)({width:"20",height:"20","aria-hidden":"true"},e),n.createElement("g",{fill:"#7a7a7a"},n.createElement("path",{d:"M9.992 10.023c0 .2-.062.399-.172.547l-4.996 7.492a.982.982 0 01-.828.454H1c-.55 0-1-.453-1-1 0-.2.059-.403.168-.551l4.629-6.942L.168 3.078A.939.939 0 010 2.528c0-.548.45-.997 1-.997h2.996c.352 0 .649.18.828.45L9.82 9.472c.11.148.172.347.172.55zm0 0"}),n.createElement("path",{d:"M19.98 10.023c0 .2-.058.399-.168.547l-4.996 7.492a.987.987 0 01-.828.454h-3c-.547 0-.996-.453-.996-1 0-.2.059-.403.168-.551l4.625-6.942-4.625-6.945a.939.939 0 01-.168-.55 1 1 0 01.996-.997h3c.348 0 .649.18.828.45l4.996 7.492c.11.148.168.347.168.55zm0 0"})))}const I="collapseSidebarButton_PEFL",A="collapseSidebarButtonIcon_kv0_";function w(e){let{onClick:t}=e;return n.createElement("button",{type:"button",title:(0,u.translate)({id:"theme.docs.sidebar.collapseButtonTitle",message:"Collapse sidebar",description:"The title attribute for collapse button of doc sidebar"}),"aria-label":(0,u.translate)({id:"theme.docs.sidebar.collapseButtonAriaLabel",message:"Collapse sidebar",description:"The title attribute for collapse button of doc sidebar"}),className:(0,l.default)("button button--secondary button--outline",I),onClick:t},n.createElement(N,{className:A}))}var y=a(65041),x=a(89532);const T=Symbol("EmptyContext"),M=n.createContext(T);function L(e){let{children:t}=e;const[a,l]=(0,n.useState)(null),r=(0,n.useMemo)((()=>({expandedItem:a,setExpandedItem:l})),[a]);return n.createElement(M.Provider,{value:r},t)}var B=a(41422),H=a(99169),P=a(75489),G=a(92303);function F(e){let{categoryLabel:t,onClick:a}=e;return n.createElement("button",{"aria-label":(0,u.translate)({id:"theme.DocSidebarItem.toggleCollapsedCategoryAriaLabel",message:"Toggle the collapsible sidebar category '{label}'",description:"The ARIA label to toggle the collapsible sidebar category"},{label:t}),type:"button",className:"clean-btn menu__caret",onClick:a})}function D(e){let{item:t,onItemClick:a,activePath:r,level:i,index:s,...d}=e;const{items:m,label:u,collapsible:b,className:p,href:h}=t,{docs:{sidebar:{autoCollapseCategories:f}}}=(0,C.p)(),E=function(e){const t=(0,G.default)();return(0,n.useMemo)((()=>e.href?e.href:!t&&e.collapsible?(0,c._o)(e):void 0),[e,t])}(t),v=(0,c.w8)(t,r),g=(0,H.ys)(h,r),{collapsed:_,setCollapsed:k}=(0,B.u)({initialState:()=>!!b&&(!v&&t.collapsed)}),{expandedItem:N,setExpandedItem:I}=function(){const e=(0,n.useContext)(M);if(e===T)throw new x.dV("DocSidebarItemsExpandedStateProvider");return e}(),A=function(e){void 0===e&&(e=!_),I(e?null:s),k(e)};return function(e){let{isActive:t,collapsed:a,updateCollapsed:l}=e;const r=(0,x.ZC)(t);(0,n.useEffect)((()=>{t&&!r&&a&&l(!1)}),[t,r,a,l])}({isActive:v,collapsed:_,updateCollapsed:A}),(0,n.useEffect)((()=>{b&&null!=N&&N!==s&&f&&k(!0)}),[b,N,s,k,f]),n.createElement("li",{className:(0,l.default)(o.G.docs.docSidebarItemCategory,o.G.docs.docSidebarItemCategoryLevel(i),"menu__list-item",{"menu__list-item--collapsed":_},p)},n.createElement("div",{className:(0,l.default)("menu__list-item-collapsible",{"menu__list-item-collapsible--active":g})},n.createElement(P.default,(0,S.A)({className:(0,l.default)("menu__link",{"menu__link--sublist":b,"menu__link--sublist-caret":!h&&b,"menu__link--active":v}),onClick:b?e=>{null==a||a(t),h?A(!1):(e.preventDefault(),A())}:()=>{null==a||a(t)},"aria-current":g?"page":void 0,"aria-expanded":b?!_:void 0,href:b?null!=E?E:"#":E},d),u),h&&b&&n.createElement(F,{categoryLabel:u,onClick:e=>{e.preventDefault(),A()}})),n.createElement(B.N,{lazy:!0,as:"ul",className:"menu__list",collapsed:_},n.createElement(O,{items:m,tabIndex:_?-1:0,onItemClick:a,activePath:r,level:i+1})))}var W=a(16654),V=a(43186);const z="menuExternalLink_NmtK";function R(e){let{item:t,onItemClick:a,activePath:r,level:i,index:s,...d}=e;const{href:m,label:u,className:b,autoAddBaseUrl:p}=t,h=(0,c.w8)(t,r),f=(0,W.A)(m);return n.createElement("li",{className:(0,l.default)(o.G.docs.docSidebarItemLink,o.G.docs.docSidebarItemLinkLevel(i),"menu__list-item",b),key:u},n.createElement(P.default,(0,S.A)({className:(0,l.default)("menu__link",!f&&z,{"menu__link--active":h}),autoAddBaseUrl:p,"aria-current":h?"page":void 0,to:m},f&&{onClick:a?()=>a(t):void 0},d),u,!f&&n.createElement(V.A,null)))}const U="menuHtmlItem_M9Kj";function j(e){let{item:t,level:a,index:r}=e;const{value:i,defaultStyle:c,className:s}=t;return n.createElement("li",{className:(0,l.default)(o.G.docs.docSidebarItemLink,o.G.docs.docSidebarItemLinkLevel(a),c&&[U,"menu__list-item"],s),key:r,dangerouslySetInnerHTML:{__html:i}})}function K(e){let{item:t,...a}=e;switch(t.type){case"category":return n.createElement(D,(0,S.A)({item:t},a));case"html":return n.createElement(j,(0,S.A)({item:t},a));default:return n.createElement(R,(0,S.A)({item:t},a))}}function q(e){let{items:t,...a}=e;return n.createElement(L,null,t.map(((e,t)=>n.createElement(K,(0,S.A)({key:t,item:e,index:t},a)))))}const O=(0,n.memo)(q),X="menu_SIkG",Y="menuWithAnnouncementBar_GW3s";function Z(e){let{path:t,sidebar:a,className:r}=e;const i=function(){const{isActive:e}=(0,y.Mj)(),[t,a]=(0,n.useState)(e);return(0,b.Mq)((t=>{let{scrollY:n}=t;e&&a(0===n)}),[e]),e&&t}();return n.createElement("nav",{"aria-label":(0,u.translate)({id:"theme.docs.sidebar.navAriaLabel",message:"Docs sidebar",description:"The ARIA label for the sidebar navigation"}),className:(0,l.default)("menu thin-scrollbar",X,i&&Y,r)},n.createElement("ul",{className:(0,l.default)(o.G.docs.docSidebarMenu,"menu__list")},n.createElement(O,{items:a,activePath:t,level:1})))}const $="sidebar_njMd",J="sidebarWithHideableNavbar_wUlq",Q="sidebarHidden_VK0M",ee="sidebarLogo_isFc";function te(e){let{path:t,sidebar:a,onCollapse:r,isHidden:o}=e;const{navbar:{hideOnScroll:i},docs:{sidebar:{hideable:c}}}=(0,C.p)();return n.createElement("div",{className:(0,l.default)($,i&&J,o&&Q)},i&&n.createElement(k.A,{tabIndex:-1,className:ee}),n.createElement(Z,{path:t,sidebar:a}),c&&n.createElement(w,{onClick:r}))}const ae=n.memo(te);var ne=a(75600),le=a(22069);const re=e=>{let{sidebar:t,path:a}=e;const r=(0,le.M)();return n.createElement("ul",{className:(0,l.default)(o.G.docs.docSidebarMenu,"menu__list")},n.createElement(O,{items:t,activePath:a,onItemClick:e=>{"category"===e.type&&e.href&&r.toggle(),"link"===e.type&&r.toggle()},level:1}))};function oe(e){return n.createElement(ne.GX,{component:re,props:e})}const ie=n.memo(oe);function ce(e){const t=(0,_.l)(),a="desktop"===t||"ssr"===t,l="mobile"===t;return n.createElement(n.Fragment,null,a&&n.createElement(ae,e),l&&n.createElement(ie,e))}const se="expandButton_m80_",de="expandButtonIcon_BlDH";function me(e){let{toggleSidebar:t}=e;return n.createElement("div",{className:se,title:(0,u.translate)({id:"theme.docs.sidebar.expandButtonTitle",message:"Expand sidebar",description:"The ARIA label and title attribute for expand button of doc sidebar"}),"aria-label":(0,u.translate)({id:"theme.docs.sidebar.expandButtonAriaLabel",message:"Expand sidebar",description:"The ARIA label and title attribute for expand button of doc sidebar"}),tabIndex:0,role:"button",onKeyDown:t,onClick:t},n.createElement(N,{className:de}))}const ue={docSidebarContainer:"docSidebarContainer_b6E3",docSidebarContainerHidden:"docSidebarContainerHidden_b3ry",sidebarViewport:"sidebarViewport_Xe31"};function be(e){var t;let{children:a}=e;const l=(0,d.t)();return n.createElement(n.Fragment,{key:null!=(t=null==l?void 0:l.name)?t:"noSidebar"},a)}function pe(e){let{sidebar:t,hiddenSidebarContainer:a,setHiddenSidebarContainer:r}=e;const{pathname:i}=(0,g.zy)(),[c,s]=(0,n.useState)(!1),d=(0,n.useCallback)((()=>{c&&s(!1),!c&&(0,v.O)()&&s(!0),r((e=>!e))}),[r,c]);return n.createElement("aside",{className:(0,l.default)(o.G.docs.docSidebarContainer,ue.docSidebarContainer,a&&ue.docSidebarContainerHidden),onTransitionEnd:e=>{e.currentTarget.classList.contains(ue.docSidebarContainer)&&a&&s(!0)}},n.createElement(be,null,n.createElement("div",{className:(0,l.default)(ue.sidebarViewport,c&&ue.sidebarViewportHidden)},n.createElement(ce,{sidebar:t,path:i,onCollapse:d,isHidden:c}),c&&n.createElement(me,{toggleSidebar:d}))))}const he={docMainContainer:"docMainContainer_gTbr",docMainContainerEnhanced:"docMainContainerEnhanced_Uz_u",docItemWrapperEnhanced:"docItemWrapperEnhanced_czyv"};function fe(e){let{hiddenSidebarContainer:t,children:a}=e;const r=(0,d.t)();return n.createElement("main",{className:(0,l.default)(he.docMainContainer,(t||!r)&&he.docMainContainerEnhanced)},n.createElement("div",{className:(0,l.default)("container padding-top--md padding-bottom--lg",he.docItemWrapper,t&&he.docItemWrapperEnhanced)},a))}const Ee="docPage__5DB",ve="docsWrapper_BCFX";function ge(e){let{children:t}=e;const a=(0,d.t)(),[l,r]=(0,n.useState)(!1);return n.createElement(m.A,{wrapperClassName:ve},n.createElement(E,null),n.createElement("div",{className:Ee},a&&n.createElement(pe,{sidebar:a.items,hiddenSidebarContainer:l,setHiddenSidebarContainer:r}),n.createElement(fe,{hiddenSidebarContainer:l},t)))}var _e=a(81774),Ce=a(41463);function ke(e){const{versionMetadata:t}=e;return n.createElement(n.Fragment,null,n.createElement(Ce.A,{version:t.version,tag:(0,i.tU)(t.pluginId,t.version)}),n.createElement(r.be,null,t.noIndex&&n.createElement("meta",{name:"robots",content:"noindex, nofollow"})))}function Se(e){const{versionMetadata:t}=e,a=(0,c.mz)(e);if(!a)return n.createElement(_e.default,null);const{docElement:i,sidebarName:m,sidebarItems:u}=a;return n.createElement(n.Fragment,null,n.createElement(ke,e),n.createElement(r.e3,{className:(0,l.default)(o.G.wrapper.docsPages,o.G.page.docsDocPage,e.versionMetadata.className)},n.createElement(s.n,{version:t},n.createElement(d.V,{name:m,items:u},n.createElement(ge,null,i)))))}},81774:(e,t,a)=>{a.r(t),a.d(t,{default:()=>i});var n=a(96540),l=a(21312),r=a(69024),o=a(68315);function i(){return n.createElement(n.Fragment,null,n.createElement(r.be,{title:(0,l.translate)({id:"theme.NotFound.title",message:"Page Not Found"})}),n.createElement(o.A,null,n.createElement("main",{className:"container margin-vert--xl"},n.createElement("div",{className:"row"},n.createElement("div",{className:"col col--6 col--offset-3"},n.createElement("h1",{className:"hero__title"},n.createElement(l.default,{id:"theme.NotFound.title",description:"The title of the 404 page"},"Page Not Found")),n.createElement("p",null,n.createElement(l.default,{id:"theme.NotFound.p1",description:"The first paragraph of the 404 page"},"We could not find what you were looking for.")),n.createElement("p",null,n.createElement(l.default,{id:"theme.NotFound.p2",description:"The 2nd paragraph of the 404 page"},"Please contact the owner of the site that linked you to the original URL and let them know their link is broken.")))))))}},32252:(e,t,a)=>{a.d(t,{n:()=>o,r:()=>i});var n=a(96540),l=a(89532);const r=n.createContext(null);function o(e){let{children:t,version:a}=e;return n.createElement(r.Provider,{value:a},t)}function i(){const e=(0,n.useContext)(r);if(null===e)throw new l.dV("DocsVersionProvider");return e}}}]);