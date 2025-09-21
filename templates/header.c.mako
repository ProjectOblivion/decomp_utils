// SPDX-License-Identifier: AGPL-3.0-or-later
#include "${ovl_header_path}"
<%
if len(header_syms) <= 12:
    overlay_type = "AbbreviatedOverlay"
# psp will probably be a length of 14 for this condition
elif len(header_syms) == 13 or len(header_syms) == 14:
    overlay_type = "u_long*"
else:
    overlay_type = "Overlay"
%>
% if ovl_type == "stage" or ovl_type == "boss":
extern RoomHeader OVL_EXPORT(rooms)[];
extern SpriteParts* OVL_EXPORT(spriteBanks)[];
extern u_long* OVL_EXPORT(cluts)[];
extern LayoutEntity* ${header_syms[7]}[];
extern RoomDef OVL_EXPORT(rooms_layers)[];
extern GfxBank* OVL_EXPORT(gfxBanks)[];
void UpdateStageEntities(void);

${overlay_type} OVL_EXPORT(Overlay) = {
    .Update = ${header_syms[0]},
    .HitDetection = ${header_syms[1]},
    .UpdateRoomPosition = ${header_syms[2]},
    .InitRoomEntities = ${header_syms[3]},
    .rooms = ${header_syms[4]},
    .spriteBanks = ${header_syms[5]},
    .cluts = ${header_syms[6]},
    .objLayoutHorizontal = ${header_syms[7]},
    .tileLayers = ${header_syms[8]},
    .gfxBanks = ${header_syms[9]},
    .UpdateStageEntities = ${header_syms[10]},
% if len(header_syms) > 12:
    .unk2C = ${header_syms[11]},
    .unk30 = ${header_syms[12]},
% endif
% if len(header_syms) > 15:
    .unk34 = ${header_syms[13]},
    .unk38 = ${header_syms[14]},
    .StageEndCutScene = ${header_syms[15]},
% endif
};
% elif ovl_type == "weapon":
Weapon OVL_EXPORT(Overlay) = {
    ${header_syms[0]},
    (void (*)(Entity*))${header_syms[1]},
    (void (*)(Entity*))${header_syms[2]},
    (void (*)(Entity*))${header_syms[3]},
    ${header_syms[4]},
    (void (*)(Entity*))${header_syms[5]},
    ${header_syms[6]},
    ${header_syms[7]},
    (void (*)(Entity*))${header_syms[8]},
    (void (*)(Entity*))${header_syms[9]},
    (void (*)(Entity*))${header_syms[10]},
    ${header_syms[11]},
    ${header_syms[12]},
    ${header_syms[13]},
    ${header_syms[14]},
    ${header_syms[15]},
};
% else:
// No header for ${ovl_type}
% endif