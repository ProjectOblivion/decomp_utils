<%
maxlen = max(len(e_init) for function, e_init in e_inits) + 1
%>
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "stage.h"

#define OVL_EXPORT(x) ${ovl_name}_##x

enum OVL_EXPORT(Entities) {
    E_NONE,
% for function, e_init in e_inits:
    ${(e_init + ",").ljust(maxlen)} // ${function}
% endfor
    NUM_ENTITIES,
};
