from collections.abc import Sequence

from avalanche.benchmarks.scenarios.dataset_scenario import DatasetExperience, TCLDataset
from avalanche.benchmarks.scenarios.generic_scenario import CLScenario, EagerCLStream
from avalanche.benchmarks.utils.data import AvalancheDataset


class MutableDatasetExperience(DatasetExperience[TCLDataset]):
    @DatasetExperience.dataset.setter
    def dataset(self, dataset: TCLDataset) -> None:
        self._dataset = dataset


def benchmark_from_datasets_mutable(**dataset_streams: Sequence[TCLDataset]) -> CLScenario:
    streams = []
    for stream_name, datasets in dataset_streams.items():
        for dataset in datasets:
            if not isinstance(dataset, AvalancheDataset):
                raise ValueError("datasets must be AvalancheDatasets")
        experiences = [
            MutableDatasetExperience(dataset=dataset, current_experience=experience_id)
            for experience_id, dataset in enumerate(datasets)
        ]
        streams.append(EagerCLStream(stream_name, experiences))
    return CLScenario(streams)
