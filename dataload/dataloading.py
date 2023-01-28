from pathlib import Path

import yaml


class DataFilesRegistry:
    """Given a data root directory, this class is able to load any data files that sits inside it.
    Datafiles are expected to be in the form of <datafile_x>.yml and to contain a list of paragraphs.

    This abstraction allows the datafile users to refer to a datafile by its name (for instance <datafile_x>) and
    to avoid referring to the full path of the datafile: This allows relocation of datafile structures without
    changing the code or other configuration files. This allows also to work in multiple environments (dev, local, prod)
    """

    def __init__(self, data_root: Path):
        if isinstance(data_root, str):
            data_root = Path(data_root)
        self.data_root = data_root
        self.glob = "**/*.yml"
        self._datafiles = {}

        for p in data_root.glob(self.glob):
            if p.stem in self._datafiles:
                raise ValueError(f"Duplicate data file with stem name: {p.stem}")
            self._datafiles[p.stem] = p

    def load_items(self, key: str) -> list[str]:
        """Loads all the paragraphs from a data file named <key>.yml that sits in any sub-folder of the data root"""
        file = self[key]
        if not self[key] or not self[key].suffix == ".yml":
            raise ValueError(f"Invalid data file: {file}")
        with open(file, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, key: str) -> Path:
        return self._datafiles[key]

    def __iter__(self):
        return iter(self._datafiles)

    def __len__(self):
        return len(self._datafiles)

    def __contains__(self, key: str) -> bool:
        return key in self._datafiles

    def __repr__(self):
        return f"DataFilesRegistry({self.data_root}, {self.glob})"

    def keys(self):
        return self._datafiles.keys()

    def values(self):
        return self._datafiles.values()

    def items(self):
        return self._datafiles.items()

    def __str__(self):
        items = ", ".join(f"{k} -> {v}" for k, v in self.items())
        return f"DataFilesRegistry({self.data_root}, {self.glob}, items={items})"
