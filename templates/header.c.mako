// SPDX-License-Identifier: AGPL-3.0-or-later
#include "${ovl_header_path}"

extern RoomHeader OVL_EXPORT(rooms)[];
extern SpriteParts* OVL_EXPORT(spriteBanks)[];
extern u_long* OVL_EXPORT(cluts)[];
extern LayoutEntity* ${header_syms[7]}[];
extern RoomDef OVL_EXPORT(rooms_layers)[];
extern GfxBank* OVL_EXPORT(gfxBanks)[];
void UpdateStageEntities(void);
<%
if len(header_syms) <= 12:
    overlay_type = "AbbreviatedOverlay"
# psp will probably be a length of 14 for this condition
elif len(header_syms) == 13:
    overlay_type = "u_long*"
else:
    overlay_type = "Overlay"
%>
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

// #include "gen/sprite_banks.h"
// #include "gen/palette_def.h"
// #include "gen/layers.h"
// #include "gen/graphics_banks.h"
