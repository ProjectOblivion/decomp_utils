// SPDX-License-Identifier: AGPL-3.0-or-later
% if ovl_type == "weapon":
#include <weapon.h>
% else:
#include <stage.h>
% endif

#define OVL_EXPORT(x) ${ovl_name.upper()}_##x
% if e_inits != None and ovl_type != "weapon":
<%
maxlen = max(len(e_init) for function, e_init in e_inits) + 1
%>
enum OVL_EXPORT(Entities) {
    E_NONE,
% for function, e_init in e_inits:
    ${(e_init + ",").ljust(maxlen)} // ${function}
% endfor
    NUM_ENTITIES,
};
% endif
