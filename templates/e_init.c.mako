// SPDX-License-Identifier: AGPL-3.0-or-later
#include "${ovl_name}.h"

% for function in entity_funcs:
void ${function}(Entity* self);
% endfor

PfnEntityUpdate OVL_EXPORT(EntityUpdates)[] = {
% for function in entity_funcs:
    ${function},
% endfor
};
