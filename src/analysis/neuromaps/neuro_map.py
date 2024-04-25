import os
import pickle
import pandas as pd
import numpy as np
from neuromaps import transforms
from neuromaps.datasets import fetch_annotation
from src.basic.constants import NEUROMAPS_DATA_DIR, DESIKAN_NEUROMAPS_DIR, NEUROMAPS_SRC_DIR
from src.utils.neuromaps_utils import convert_to_desikan, get_mapping_labels

VOLUMETRIC_SPACES = ['MNI152']
SURFACE_SPACES = ['fsaverage', 'fsLR', 'civet']


class NeuroMap:
    """
    This class encapsulate neuro maps.
    """

    def __init__(self,
                 info: pd.Series,
                 convert_to_desikan: bool = True,
                 intermediate_space: list[str, str] | tuple[str, str] = None):
        """
        Initialize the class.

        Args:
            info: Information about the neuro map, which includes columns: ['annotation', 'description',
                'tags', 'N (males)', 'age (years)', 'primary reference(s)', 'secondary reference(s)']
            data_path: Path to the data file.
        """
        self.info: pd.Series = info
        self.annotation: tuple[str, str, str,
                               str] = eval(self.info["annotation"])
        self.intermediate_space = intermediate_space
        if self.intermediate_space is None:
            self.intermediate_space = ['fsaverage', '10k']
        self.target_space, self.target_resolution = self.intermediate_space

        self.data_path: str | list[str, str] = fetch_annotation(
            source=self.source,
            desc=self.descriptor,
            space=self.space,
            res=self.resolution,
            data_dir=NEUROMAPS_DATA_DIR)

        # if the space is one of the surface spaces, and the data_path is only a string,
        # then we need to duplicate the hemisphere
        if self.is_surface and isinstance(self.data_path, str):
            self.data_path = [self.data_path, self.data_path]

        if convert_to_desikan:
            self.convert_to_desikan(save_converted=True)

    @property
    def description(self) -> str:
        """
        Get the description of the neuro map.
        """
        return self.info["description"]

    @property
    def tags(self) -> str:
        """
        Get the tags of the neuro map.
        """
        return self.info["tags"]

    @property
    def source(self) -> str:
        """
        Get the source of the neuro map.
        """
        return self.annotation[0]

    @property
    def descriptor(self) -> str:
        """
        Get the descriptor of the neuro map.
        """
        return self.annotation[1]

    @property
    def space(self) -> str:
        """
        Get the space of the neuro map.
        """
        return self.annotation[2]

    @property
    def resolution(self) -> str:
        """
        Get the resolution of the neuro map.
        """
        return self.annotation[3]

    @property
    def is_volumetric(self) -> bool:
        """
        Check if the neuro map is in a volumetric space.
        """
        return self.space in VOLUMETRIC_SPACES

    @property
    def is_surface(self) -> bool:
        """
        Check if the neuro map is in a surface space.
        """
        return self.space in SURFACE_SPACES

    def convert(self, save_converted: bool = True) -> None:
        """
        Convert the neuro map to the intermediate space and resolution.
        """

        convert_func_name = f"{self.space.lower()}_to_{self.target_space.lower()}"
        convert_func = getattr(transforms, convert_func_name)
        self.intermediate_map = convert_func(self.data_path,
                                             self.target_resolution)

        self.intermediate_data = self.get_intermediate_data()

        if save_converted:
            self.save_intermediate_map()

        print(f"{self} converted to {self.intermediate_space}.")

        return self.intermediate_map

    def save_intermediate_map(self,
                              save_dir: str = NEUROMAPS_DATA_DIR) -> None:
        """
        Save the converted neuro map to a csv file.
        """
        file_name = '_'.join([self.source, self.descriptor])
        subfolder_name = f"{self.target_space}_{self.target_resolution}"
        save_dir = os.path.join(save_dir, subfolder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{file_name}.csv")
        pd.DataFrame(self.intermediate_data).to_csv(file_path,
                                                    header=False,
                                                    index=False)

    def get_intermediate_lh_data(self) -> np.ndarray:
        """
        Get the converted data in the left hemisphere.
        """
        if self.target_space in VOLUMETRIC_SPACES:
            raise ValueError(
                "The converted data is in volumetric space. Hence, it does not store left and right hemispheres as separate objects"
            )
        return self.intermediate_map[0].agg_data()

    def get_intermediate_rh_data(self) -> np.ndarray:
        """
        Get the converted data in the right hemisphere.
        """
        if self.target_space in VOLUMETRIC_SPACES:
            raise ValueError(
                "The converted data is in volumetric space. Hence, it does not store left and right hemispheres as separate objects"
            )
        return self.intermediate_map[1].agg_data()

    def get_intermediate_data(self) -> np.ndarray:
        """
        Get the converted data in the left and right hemisphere.
        """
        data = np.concatenate(
            [self.get_intermediate_lh_data(),
             self.get_intermediate_rh_data()],
            axis=0)
        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def convert_to_desikan(self, save_converted: bool = True) -> np.ndarray:
        """
        Convert the converted neuro map to Desikan-Killiany atlas.
        The default intermediate space is fsaverage 10k (fsaverage6).
        """

        if not hasattr(self, 'intermediate_map'):
            self.convert(save_converted=save_converted)

        self.map_in_desikan = convert_to_desikan(
            self.intermediate_data,
            get_mapping_labels(*self.intermediate_space))

        if save_converted:
            self.save_map_in_desikan()

        print(f"{self} converted to the Desikan-Killiany atlas.")

        return self.map_in_desikan

    def save_map_in_desikan(self,
                            save_dir: str = DESIKAN_NEUROMAPS_DIR) -> None:
        """
        Save the converted neuro map in Desikan-Killiany atlas to a csv file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        file_name = '_'.join([self.source, self.descriptor])
        file_path = os.path.join(save_dir, f"{file_name}.csv")
        pd.DataFrame(self.map_in_desikan).to_csv(file_path,
                                                 header=False,
                                                 index=False)

    def __str__(self):
        return f"{self.source}_{self.descriptor}"


def get_all_maps(
    space_and_res: str | list[str, str] = ['fsaverage',
                                           '10k']) -> list[NeuroMap]:
    """
    Get all the neuro maps in the Desikan-Killiany atlas.
    If all the maps are already processed and saved, we can directly read it,
    Otherwise, we will process all the maps and save them using pickle.
    """
    subfolder_name = space_and_res if isinstance(
        space_and_res, str) else '_'.join(space_and_res)
    used_neuromaps_info = pd.read_csv(
        os.path.join(NEUROMAPS_SRC_DIR, 'used_neuromaps_info.csv'))
    all_maps_file_path = os.path.join(NEUROMAPS_DATA_DIR, subfolder_name,
                                      'all_maps.pkl')

    # load the map if the file exists, and get all NeuroMaps if not
    if os.path.exists(all_maps_file_path):
        with open(all_maps_file_path, 'rb') as f:
            all_maps = pickle.load(f)
        return all_maps

    intermediate_space = space_and_res if isinstance(space_and_res,
                                                     list) else None
    all_maps = [
        NeuroMap(map_info,
                 convert_to_desikan=True,
                 intermediate_space=intermediate_space)
        for _, map_info in used_neuromaps_info.iterrows()
    ]

    with open(all_maps_file_path, 'wb') as f:
        pickle.dump(all_maps, f)

    return all_maps


def get_all_maps_data(space_and_res: str
                      | list[str, str] = ['fsaverage', '10k'],
                      save_mean_map: bool = True) -> pd.DataFrame:
    """
    Get the data of all neuro maps in the Desikan-Killiany atlas.
    If the data of all the maps are already processed and saved, we can directly read it,
    Otherwise, we will get the data of all the maps and save them to a csv.
    """
    subfolder_name = space_and_res if isinstance(
        space_and_res, str) else '_'.join(space_and_res)

    all_maps_data_file_path = os.path.join(NEUROMAPS_DATA_DIR, subfolder_name,
                                           'all_maps_data.csv')
    # load the data if the file exists, and get all the data if not
    if os.path.exists(all_maps_data_file_path):
        all_maps_data = pd.read_csv(all_maps_data_file_path)
        return all_maps_data

    all_maps = get_all_maps(space_and_res)
    if space_and_res == 'desikan':
        all_maps_data = np.stack([map.map_in_desikan for map in all_maps],
                                 axis=1)
    else:
        all_maps_data = np.stack([map.intermediate_data for map in all_maps],
                                 axis=1)

    # get headers for the csv file
    headers = [str(map) for map in all_maps]
    all_maps_data = pd.DataFrame(all_maps_data, columns=headers)
    all_maps_data.to_csv(all_maps_data_file_path, index=False)
    if save_mean_map:
        mean_map_file_path = os.path.join(NEUROMAPS_DATA_DIR, subfolder_name,
                                          'mean_map.csv')
        mean_map = all_maps_data.mean(axis=1)
        pd.DataFrame(mean_map).to_csv(mean_map_file_path,
                                      header=False,
                                      index=False)
    return all_maps_data
