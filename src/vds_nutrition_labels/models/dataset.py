from collections import Counter
from dataclasses import dataclass


@dataclass
class Sample:
    function: str
    label: int
    cve: str
    cwe: list[str]
    project: str


@dataclass
class Dataset:
    name: str
    description: str
    version: str
    license: str
    data: list[Sample] = None
    train: list[Sample] = None
    test: list[Sample] = None
    validation: list[Sample] = None


    def has_splits(self) -> bool:
        return any([self.train, self.test, self.validation])

    def summary(self) -> str:
        train = self.train or []
        test = self.test or []
        validation = self.validation or []
        data = self.data or []

        if self.has_splits():
            split_counts = {
                "train": len(train),
                "test": len(test),
                "validation": len(validation),
            }
            total_count = sum(split_counts.values())
            samples = train + test + validation
        else:
            split_counts = {"all": len(data)}
            total_count = len(data)
            samples = data

        label_counter = Counter(sample.label for sample in samples)
        label_lines = [f"      {label}: {count}" for label, count in sorted(label_counter.items())]
        sample_lines = "\n".join(f"    {name}: {count}" for name, count in split_counts.items())

        return (
            f"Dataset summary for {self.name}\n"
            f"  description: {self.description}\n"
            f"  version: {self.version}\n"
            f"  license: {self.license}\n"
            f"  samples:\n"
            f"{sample_lines}\n"
            f"    total: {total_count}\n"
            f"  label distribution:\n"
            + "\n".join(label_lines)
        )
