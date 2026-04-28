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

        train_count = len(train)
        test_count = len(test)
        valid_count = len(validation)
        total_count = train_count + test_count + valid_count

        label_counter = Counter(sample.label for sample in train + test + validation)
        label_lines = [f"      {label}: {count}" for label, count in sorted(label_counter.items())]

        return (
            f"Dataset summary for {self.name}\n"
            f"  description: {self.description}\n"
            f"  version: {self.version}\n"
            f"  license: {self.license}\n"
            f"  samples:\n"
            f"    train: {train_count}\n"
            f"    test: {test_count}\n"
            f"    validation: {valid_count}\n"
            f"    total: {total_count}\n"
            f"  label distribution:\n"
            + "\n".join(label_lines)
        )
