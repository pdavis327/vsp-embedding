# 1. Frame Styles Dictionary: Defines each target frame style along with lists of adjectives, frame types, and specifications commonly associated with their design characteristics. This helps inject subtle "signals" into the descriptions.
# 2. Random Selection: For each sample, it randomly picks a frame style, then randomly selects an adjective, frame type, and specification from that style's defined characteristics.
# 3. Description Templates: It uses a few basic sentence structures to combine these elements.
# 4. Frame-Specific Details: It then adds more specific, randomized details based on the chosen frame type (e.g., materials for metal frames, colors for plastic frames) to make the descriptions richer and more realistic.
# 5. DataFrame Creation: Finally, it compiles all the generated descriptions and frame styles into a pandas DataFrame.

import pandas as pd
import random


def generate_synthetic_eyeglass_frame_data(num_samples=500):
    """
    Generates a synthetic dataset of eyeglass frame descriptions and their
    simulated frame styles.

    Args:
        num_samples (int): The number of data samples to generate.

    Returns:
        pandas.DataFrame: A DataFrame with 'Frame_Description' and 'Frame_Style' columns.
    """

    # Define possible frame styles and their associated typical characteristics/designs
    frame_styles = {
        "Classic": {
            "adjectives": [
                "timeless",
                "traditional",
                "elegant",
                "sophisticated",
                "refined",
                "understated",
            ],
            "frame_types": [
                "round metal frames",
                "oval acetate frames",
                "rectangular metal frames",
                "wire-rim frames",
                "tortoiseshell frames",
                "gold-plated frames",
            ],
            "specs": [
                "thin wire construction",
                "adjustable nose pads",
                "spring hinges",
                "premium materials",
                "handcrafted details",
            ],
        },
        "Modern": {
            "adjectives": [
                "contemporary",
                "sleek",
                "minimalist",
                "clean-lined",
                "progressive",
                "innovative",
            ],
            "frame_types": [
                "geometric frames",
                "rimless frames",
                "semi-rimless frames",
                "bold rectangular frames",
                "angular frames",
                "monocle-style frames",
            ],
            "specs": [
                "ultra-lightweight",
                "memory metal",
                "flexible hinges",
                "anti-reflective coating",
                "blue light filtering",
            ],
        },
        "Vintage": {
            "adjectives": [
                "retro-inspired",
                "nostalgic",
                "art deco",
                "mid-century",
                "vintage-inspired",
                "classic revival",
            ],
            "frame_types": [
                "cat-eye frames",
                "horn-rimmed frames",
                "aviator frames",
                "wayfarer frames",
                "round wire frames",
                "browline frames",
            ],
            "specs": [
                "authentic period details",
                "hand-polished finish",
                "vintage hardware",
                "distressed patina",
                "period-correct sizing",
            ],
        },
        "Sporty": {
            "adjectives": [
                "athletic",
                "durable",
                "performance-oriented",
                "rugged",
                "active lifestyle",
                "adventure-ready",
            ],
            "frame_types": [
                "wraparound frames",
                "sports sunglasses",
                "cycling frames",
                "swimming goggles",
                "ski goggles",
                "protective eyewear",
            ],
            "specs": [
                "impact-resistant",
                "UV protection",
                "anti-fog coating",
                "ventilation system",
                "quick-release temples",
            ],
        },
        "Luxury": {
            "adjectives": [
                "premium",
                "exclusive",
                "high-end",
                "designer",
                "luxurious",
                "prestigious",
            ],
            "frame_types": [
                "titanium frames",
                "precious metal frames",
                "designer collaboration frames",
                "limited edition frames",
                "custom-made frames",
                "luxury brand frames",
            ],
            "specs": [
                "precious metal construction",
                "hand-engraved details",
                "genuine leather accents",
                "premium lens options",
                "custom fitting service",
            ],
        },
        "Fashion": {
            "adjectives": [
                "trendy",
                "stylish",
                "fashion-forward",
                "bold",
                "statement-making",
                "runway-inspired",
            ],
            "frame_types": [
                "oversized frames",
                "colorful acetate frames",
                "geometric shapes",
                "mixed material frames",
                "decorative frames",
                "novelty frames",
            ],
            "specs": [
                "vibrant color options",
                "patterned acetate",
                "decorative elements",
                "trendy finishes",
                "seasonal collections",
            ],
        },
        "Professional": {
            "adjectives": [
                "business-appropriate",
                "conservative",
                "professional",
                "refined",
                "corporate",
                "executive",
            ],
            "frame_types": [
                "rectangular metal frames",
                "oval acetate frames",
                "semi-rimless frames",
                "wire frames",
                "conservative shapes",
                "neutral colored frames",
            ],
            "specs": [
                "conservative sizing",
                "neutral color palette",
                "durable construction",
                "comfortable fit",
                "professional appearance",
            ],
        },
        "Youthful": {
            "adjectives": [
                "fun",
                "playful",
                "energetic",
                "colorful",
                "trendy",
                "youth-oriented",
            ],
            "frame_types": [
                "bright colored frames",
                "funky shapes",
                "cartoon-inspired frames",
                "neon colored frames",
                "patterned frames",
                "novelty designs",
            ],
            "specs": [
                "bright color options",
                "fun patterns",
                "lightweight materials",
                "comfortable fit",
                "affordable pricing",
            ],
        },
    }

    data = []
    style_list = list(frame_styles.keys())

    for _ in range(num_samples):
        # Randomly choose a frame style for this sample
        frame_style = random.choice(style_list)
        style_info = frame_styles[frame_style]

        # Select a random frame type and adjective/spec from the style's typical characteristics
        frame_type = random.choice(style_info["frame_types"])
        adjective = random.choice(style_info["adjectives"])
        spec = random.choice(style_info["specs"])

        # Create a basic description template
        description_templates = [
            f"{adjective} {frame_type}, {spec}.",
            f"A {spec} {frame_type} for {adjective} applications.",
            f"{frame_type} ({adjective}) with {spec} features.",
            f"Designed for {adjective} needs: {frame_type} with {spec}.",
        ]

        # Add more specific details based on frame type
        frame_details = ""
        if "metal" in frame_type:
            frame_details = f" {random.choice(['stainless steel', 'titanium', 'aluminum', 'gold-plated'])} construction, {random.choice(['adjustable nose pads', 'spring hinges', 'memory metal'])}."
        elif "acetate" in frame_type:
            frame_details = f" {random.choice(['tortoiseshell', 'solid black', 'tortoise brown', 'clear'])} {random.choice(['matte finish', 'glossy finish', 'textured surface'])}."
        elif "wire" in frame_type:
            frame_details = f" {random.choice(['thin wire', 'medium wire', 'thick wire'])} {random.choice(['gold-plated', 'silver-plated', 'gunmetal'])}."
        elif "rimless" in frame_type:
            frame_details = f" {random.choice(['ultra-lightweight', 'minimalist design', 'invisible frame'])} {random.choice(['titanium screws', 'nylon thread', 'wire temples'])}."
        elif "semi-rimless" in frame_type:
            frame_details = f" {random.choice(['top rim only', 'bottom rim only', 'partial rim'])} {random.choice(['nylon thread', 'wire rim', 'acetate rim'])}."
        elif "oversized" in frame_type:
            frame_details = f" {random.choice(['large lenses', 'wide temples', 'bold presence'])} {random.choice(['statement piece', 'fashion forward', 'runway inspired'])}."
        elif "cat-eye" in frame_type:
            frame_details = f" {random.choice(['vintage inspired', 'retro glamour', 'feminine touch'])} {random.choice(['upswept corners', 'pointed edges', 'curved design'])}."
        elif "aviator" in frame_type:
            frame_details = f" {random.choice(['classic pilot style', 'teardrop lenses', 'double bridge'])} {random.choice(['metal construction', 'adjustable nose pads', 'spring hinges'])}."
        elif "wayfarer" in frame_type:
            frame_details = f" {random.choice(['iconic shape', 'bold rectangular', 'timeless design'])} {random.choice(['acetate construction', 'solid colors', 'classic proportions'])}."
        elif "browline" in frame_type:
            frame_details = f" {random.choice(['vintage charm', 'distinctive brow bar', 'retro appeal'])} {random.choice(['metal brow bar', 'acetate bottom', 'classic styling'])}."

        frame_description = random.choice(description_templates) + frame_details.strip()
        data.append(
            {
                "Frame_Description": frame_description,
                "Frame_Style": frame_style,
            }
        )

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":

    # Generate a dataset with n samples
    synthetic_df = generate_synthetic_eyeglass_frame_data(num_samples=1000)
    print(synthetic_df.head())
    print("\nDataset Info:")
    print(synthetic_df.info())
    print("\nFrame Style Distribution:")
    print(synthetic_df["Frame_Style"].value_counts())

    synthetic_df.to_csv("../data/synthetic_eyeglass_frames_1k.csv", index=False)
