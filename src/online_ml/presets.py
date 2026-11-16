from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    title: str
    positive: str
    negative: str
    positive_classes: tuple[str, ...]
    negative_classes: tuple[str, ...]
    note: str = "Review every candidate: source annotations are not binary task labels."
    # Short display names for the two classes (label 1 and label 0).
    positive_name: str = "Positive"
    negative_name: str = "Negative"


PRESETS = {
    "religion": Preset(
        "Religious Symbols & Crosses",
        "religious symbol, cross, or place of worship",
        "ordinary everyday object or plain scenery",
        ("Sculpture", "Building"),
        ("Chair", "Bottle", "Tree"),
        "Open Images has no direct cross/place-of-worship class in its 600-class subset. "
        "Building and Sculpture are only search candidates, NOT religious labels. "
        "Expect to reject many candidates or import your own reviewed images. "
        "Label visible content, not a person's religious identity.",
        "Religious", "Everyday",
    ),
    "people": Preset(
        "People & Crowds",
        "person, portrait, or crowd of people",
        "empty landscape or object without people",
        ("Person",),
        ("Tree", "Chair", "Bottle"),
        positive_name="People", negative_name="No people",
    ),
    "animals": Preset(
        "Wild & Domestic Animals",
        "animal, pet, bird, or wildlife",
        "plants, buildings, or inanimate objects",
        ("Dog", "Cat", "Bird", "Horse"),
        ("Tree", "Building", "Chair"),
        positive_name="Animal", negative_name="No animal",
    ),
    "nature": Preset(
        "Scenic Nature",
        "natural landscape, forest, mountain, or water",
        "indoor room, city street, or human portrait",
        ("Tree", "Palm tree"),
        ("Office building", "Human face", "Couch"),
        "Trees are retrieval candidates, not proof of a scenic landscape. "
        "Review scene context and import additional mountain/water scenes if needed.",
        "Nature", "Not nature",
    ),
    "pets": Preset(
        "Pets vs Wild Animals",
        "pet dog or cat",
        "wild animal or bird in nature",
        ("Dog", "Cat"),
        ("Bird", "Deer", "Squirrel", "Fox", "Lizard"),
        "Farm animals (horses, donkeys, cattle) are neither class; skip them.",
        "Pet", "Wild",
    ),
    "scenery": Preset(
        "Natural vs Landscaped",
        "wild natural landscape, forest, coast, or river",
        "landscaped resort, swimming pool, or garden",
        ("Tree",),
        ("Swimming pool", "Flowerpot"),
        "Palm trees and greenery appear on both sides. Label whether the scene is "
        "wild nature or a designed space; skip plant close-ups and interiors.",
        "Natural", "Landscaped",
    ),
}

PROMPT_TEMPLATES = ("{}", "a photo of {}", "an image showing {}")
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
