from typing import TYPE_CHECKING
import wx
import os

from amulet.api.selection import SelectionGroup
from amulet.api.errors import ChunkLoadError
from amulet.api.data_types import Dimension, OperationReturnType
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper

from amulet_map_editor.api.wx.ui.version_select import VersionSelect
from amulet_map_editor.programs.edit.api.operations import (
    SimpleOperationPanel,
    OperationError,
)

if TYPE_CHECKING:
    from amulet.api.level import BaseLevel
    from amulet_map_editor.programs.edit.api.canvas import EditCanvas

class ExportSpongeSchematic(SimpleOperationPanel):
    def __init__(
        self,
        parent: wx.Window,
        canvas: "EditCanvas",
        world: "BaseLevel",
        options_path: str,
    ):
        SimpleOperationPanel.__init__(self, parent, canvas, world, options_path)

        options = self._load_options({})

        self._dir_picker = wx.DirPickerCtrl(
            self,
            path=options.get("path", ""),
            message="Select output directory",
            style=wx.DIRP_USE_TEXTCTRL,
        )
        self._sizer.Add(self._dir_picker, 0, wx.ALL | wx.CENTER, 5)

        categories = {
            "Size": ['tiny', 'small', 'medium', 'large', 'huge'],
            "Building Type": ['well', 'townhall', 'house', 'stalls', 'gazebo', 'mansion', 'outpost', 'windmill', 'church', 'inn', 'market', 'lighthouse', 'barn', 'stable', 'farm', 'forge', 'blacksmith', 'tavern', 'temple', 'shrine', 'monument', 'statue', 'arena', 'theater', 'colosseum', 'palace', 'castle', 'fortress', 'keep', 'dungeon', 'prison', 'tower', 'laboratory', 'observatory', 'library', 'academy', 'school', 'guildhall'],
            "Building Shape": ['rectangle', 'circle', 'slanted', 'symmetric', 'curved', 'square', 'round'],
            "Environment": ['flatground', 'hill', 'overgrown', 'water'],
            "Material": ['wool', 'hay', 'stone', 'log', 'dirt'],
            "Architecture": ['vaulted', 'tiered', 'stilted', 'porch', 'ceiling', 'floor', 'attic', 'chimney'],
            "Floors": ['onefloor', 'twofloor', 'threefloor', 'fourfloor', 'fivefloor', 'sixfloor', 'sevenfloor', 'eightfloor', 'ninefloor', 'tenfloor'],
            "Other": ['detailed', 'simple', 'high', 'lookout', 'overhang', 'docks', 'sideways', 'open']
        }

         # Create a scrolled window
        scrollwin = wx.ScrolledWindow(self, -1, size=(300, 1200), style=wx.VSCROLL | wx.HSCROLL)
        scrollwin.SetScrollRate(5, 5)
        scrollwin_sizer = wx.BoxSizer(wx.VERTICAL)

        self._checkboxes = {}
        for category, labels in categories.items():
            category_sizer = wx.BoxSizer(wx.HORIZONTAL)
            category_label_sizer = wx.BoxSizer(wx.VERTICAL)
            category_checkbox_sizer = wx.BoxSizer(wx.VERTICAL)

            category_label = wx.StaticText(scrollwin, label=category, style=wx.BOLD)
            category_label_sizer.Add(category_label, 0, wx.ALL | wx.CENTER, 5)
            category_sizer.Add(category_label_sizer, 0, wx.EXPAND)

            for label in labels:
                checkbox = wx.CheckBox(scrollwin, label=label)
                category_checkbox_sizer.Add(checkbox, 0, wx.ALL | wx.CENTER, 5)
                self._checkboxes[label] = checkbox

            category_sizer.Add(category_checkbox_sizer, 1, wx.EXPAND)
            scrollwin_sizer.Add(category_sizer, 0, wx.EXPAND | wx.ALL, 5)

        scrollwin.SetSizer(scrollwin_sizer)
        self._sizer.Add(scrollwin, 1, wx.EXPAND)

        self._version_define = VersionSelect(
            self,
            world.translation_manager,
            options.get("platform", None) or world.level_wrapper.platform,
            allowed_platforms=("java",),
            allow_numerical=False,
        )
        self._sizer.Add(self._version_define, 0, wx.CENTRE, 5)
        self._add_run_button("Export")
        self.Layout()

    def resize(self):
        """Resize the parent window to accommodate this operation's UI."""
        # Get the current size of the parent window
        width, height = self.GetSize()

        # Adjust the size as needed
        self.SetSize((width, height+200))  # or whatever size you need

    def disable(self):
        self._save_options(
            {
                "path": self._dir_picker.GetPath(),
                "version": self._version_define.version_number,
            }
        )

    def _operation(
        self, world: "BaseLevel", dimension: Dimension, selection: SelectionGroup
    ) -> OperationReturnType:
        if len(selection.selection_boxes) == 0:
            raise OperationError("No selection was given to export.")
        elif len(selection.selection_boxes) != 1:
            raise OperationError(
                "The Sponge Schematic format only supports a single selection box."
            )

        labels = [label for label, checkbox in self._checkboxes.items() if checkbox.IsChecked()]

        filename = "_".join(labels) if labels else "default"
        filename = filename + '.schem'
        path = os.path.join(self._dir_picker.GetPath(), filename)

        if os.path.exists(path):
            base = os.path.splitext(path)[0]
            index = 1
            while os.path.exists(f"{base}_{index}.schem"):
                index += 1
            path = f"{base}_{index}.schem"

        version = self._version_define.version_number
        if isinstance(path, str):
            wrapper = SpongeSchemFormatWrapper(path)
            if wrapper.exists:
                response = wx.MessageDialog(
                    self,
                    f"A file is already present at {path}. Do you want to continue?",
                    style=wx.YES | wx.NO,
                ).ShowModal()
                if response == wx.ID_CANCEL:
                    return
            wrapper.create_and_open("java", version, selection, True)
            wrapper.translation_manager = world.translation_manager
            wrapper_dimension = wrapper.dimensions[0]
            chunk_count = len(list(selection.chunk_locations()))
            yield 0, f"Exporting {os.path.basename(path)}"
            for chunk_index, (cx, cz) in enumerate(selection.chunk_locations()):
                try:
                    chunk = world.get_chunk(cx, cz, dimension)
                    wrapper.commit_chunk(chunk, wrapper_dimension)
                except ChunkLoadError:
                    continue
                yield (chunk_index + 1) / chunk_count
            wrapper.save()
            wrapper.close()
        else:
            raise OperationError(
                "Please specify a save location and version in the options before running."
            )

export = {
    "name": "Export Sponge Schematic",  
    "operation": ExportSpongeSchematic,  
}
