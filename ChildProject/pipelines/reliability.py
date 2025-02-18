import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.pipeline import Pipeline


class ReliabilityAnalysis:
    def __init__(
        self,
        project: ChildProject,
        reference_set: str,
        hypothesis_set: str,
        speakers=None,
        granularity=10,
    ):
        """
        Initializes the reliability analysis with multilabel support.

        Args:
            project (ChildProject): The ChildProject object.
            reference_set (str): Name of the reference annotation set.
            hypothesis_set (str): Name of the hypothesis annotation set.
            speakers (list, optional): List of speakers to include in the analysis. Defaults to ['CHI', 'OCH', 'FEM', 'MAL'].
            granularity (int): Temporal resolution of the grid in milliseconds.
        """
        self.project = project
        self.reference_set = reference_set
        self.hypothesis_set = hypothesis_set
        self.speakers = (
            speakers if speakers is not None else ["CHI", "OCH", "FEM", "MAL"]
        )
        self.granularity = granularity

    def load_data(self):
        """
        Loads and filters annotation data for the specified speakers.
        """
        am = AnnotationManager(self.project)
        intersection = AnnotationManager.intersection(
            am.annotations, [self.reference_set, self.hypothesis_set]
        )

        if intersection.empty:
            raise ValueError(
                f"Reference set '{self.reference_set}' or hypothesis set '{self.hypothesis_set}' not found in annotations."
            )

        segments = am.get_collapsed_segments(intersection)

        if segments.empty:
            raise ValueError(
                f"No valid segments found for reference set '{self.reference_set}' and hypothesis set '{self.hypothesis_set}'."
            )

        segments = segments[segments["speaker_type"].isin(self.speakers)]

        return segments

    def segments_to_multilabel_grid(self, segments, annotation_set):
        """
        Convert segments to a multilabel grid with the specified granularity.

        Args:
            segments (pd.DataFrame): Segments data.
            annotation_set (str): The annotation set to process.

        Returns:
            np.array: A multilabel grid representation.
        """

        if segments.empty:
            return np.empty(0, dtype=object)

        if annotation_set not in segments["set"].unique():
            raise ValueError(
                f"Annotation set '{annotation_set}' not found in the data."
            )

        required_columns = {"set", "segment_onset", "segment_offset", "speaker_type"}
        missing_columns = required_columns - set(segments.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        if (segments["segment_onset"] > segments["segment_offset"]).any():
            raise ValueError(
                "Invalid segment times: 'segment_onset' must be ≤ 'segment_offset'."
            )

        max_offset = segments["segment_offset"].max()
        grid_points = int(np.ceil(max_offset / self.granularity))

        # Initialize a multilabel grid as an array of sets
        grid = np.empty((grid_points,), dtype=object)
        for i in range(grid_points):
            grid[i] = set()

        for _, row in segments[segments["set"] == annotation_set].iterrows():
            start = int(row["segment_onset"] / self.granularity)
            end = min(int(row["segment_offset"] / self.granularity), grid_points - 1)

            for i in range(start, end + 1):
                if row["speaker_type"] in self.speakers:
                    grid[i].add(row["speaker_type"])

        return grid

    def generate_confusion_matrices(self, ref_grid, hyp_grid, output_directory):
        """
        Generates and saves multiclass confusion matrices, both raw and normalized.

        Args:
            ref_grid (np.array): Reference multilabel grid.
            hyp_grid (np.array): Hypothesis multilabel grid.
            output_directory (str): Directory to save the confusion matrices.
        """
        # Replace '/' with '-' in set names to avoid errors in file names
        ref_set_safe = self.reference_set.replace("/", "-")
        hyp_set_safe = self.hypothesis_set.replace("/", "-")

        # Possible classes for the confusion matrix
        classes = self.speakers + ["SILENCE"]

        # Convert each point in ref_grid and hyp_grid into a unique label based on present speakers
        ref_labels = []
        hyp_labels = []

        for ref_set, hyp_set in zip(ref_grid, hyp_grid):
            ref_label = "-".join(sorted(ref_set)) if ref_set else "SILENCE"
            hyp_label = "-".join(sorted(hyp_set)) if hyp_set else "SILENCE"
            ref_labels.append(ref_label)
            hyp_labels.append(hyp_label)

        # Generate the raw confusion matrix
        conf_matrix = confusion_matrix(ref_labels, hyp_labels, labels=classes)

        # Create a DataFrame for the raw matrix
        conf_matrix_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
        conf_matrix_df.to_csv(
            os.path.join(
                output_directory,
                f"confusion_matrix_{ref_set_safe}_vs_{hyp_set_safe}.csv",
            )
        )

        # Visualize and save the raw matrix
        plt.figure(figsize=(10, 8))
        im = plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        # sns.heatmap(conf_matrix_df, annot=True, cmap="Blues", fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix ({self.reference_set} vs {self.hypothesis_set})")
        plt.colorbar(im)

        # Add labels
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = conf_matrix.max() / 2.0
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

        plt.xlabel("Hypothesis")
        plt.ylabel("Reference")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_directory,
                f"confusion_matrix_{ref_set_safe}_vs_{hyp_set_safe}.jpg",
            )
        )
        plt.close()

        # Normalized matrix
        normalized_conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        normalized_conf_matrix_df = pd.DataFrame(
            normalized_conf_matrix, index=classes, columns=classes
        )
        normalized_conf_matrix_df.to_csv(
            os.path.join(
                output_directory,
                f"normalized_confusion_matrix_{ref_set_safe}_vs_{hyp_set_safe}.csv",
            )
        )

        # Visualize normalized matrix
        plt.figure(figsize=(10, 8))
        im = plt.imshow(
            normalized_conf_matrix, interpolation="nearest", cmap=plt.cm.Reds
        )
        plt.title(
            f"Normalized Confusion Matrix ({self.reference_set} vs {self.hypothesis_set})"
        )
        plt.colorbar(im)

        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        fmt = ".2f"
        thresh = normalized_conf_matrix.max() / 2.0
        for i, j in np.ndindex(normalized_conf_matrix.shape):
            plt.text(
                j,
                i,
                format(normalized_conf_matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if normalized_conf_matrix[i, j] > thresh else "black",
            )

        plt.xlabel("Hypothesis")
        plt.ylabel("Reference")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_directory,
                f"normalized_confusion_matrix_{ref_set_safe}_vs_{hyp_set_safe}.jpg",
            )
        )
        plt.close()

    def compute_classification_report(self, ref_grid, hyp_grid, output_directory):
        """
        Computes and saves a classification report with Precision, Recall, and F-measure.

        Args:
            ref_grid (np.array): Reference multilabel grid.
            hyp_grid (np.array): Hypothesis multilabel grid.
            output_directory (str): Directory to save the report.
        """
        # Convert each entry in the grids to a binary vector for each speaker
        ref_binary = np.array(
            [
                [1 if speaker in labels else 0 for speaker in self.speakers]
                for labels in ref_grid
            ]
        )
        hyp_binary = np.array(
            [
                [1 if speaker in labels else 0 for speaker in self.speakers]
                for labels in hyp_grid
            ]
        )

        ref_set_safe = self.reference_set.replace("/", "-")
        hyp_set_safe = self.hypothesis_set.replace("/", "-")

        # Ensure both ref_binary and hyp_binary have the same shape
        if ref_binary.shape != hyp_binary.shape:
            raise ValueError(
                "Mismatch in binary label shapes between reference and hypothesis grids."
            )

        # Compute precision, recall, F-score, and support for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            ref_binary, hyp_binary, average=None, zero_division=0
        )
        overall_precision, overall_recall, overall_f1, _ = (
            precision_recall_fscore_support(
                ref_binary, hyp_binary, average="weighted", zero_division=0
            )
        )

        # Create a DataFrame for the detailed report
        report_df = pd.DataFrame(
            {
                "Speaker": self.speakers,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Support": support,
            }
        )

        # Add overall metrics to the report
        overall_row = pd.DataFrame(
            [["Overall", overall_precision, overall_recall, overall_f1, support.sum()]],
            columns=report_df.columns,
        )
        report_df = pd.concat([report_df, overall_row], ignore_index=True)

        # Save report to CSV
        report_df.to_csv(
            os.path.join(
                output_directory,
                f"classification_report_{ref_set_safe}_vs_{hyp_set_safe}.csv",
            ),
            index=False,
        )

    def extract(self, output_directory: str):
        """
        Runs the multilabel reliability analysis and saves the results.

        Args:
            output_directory (str): Path to the directory where results will be saved.
        """
        os.makedirs(output_directory, exist_ok=True)

        # Load data
        segments = self.load_data()

        # Generate multilabel grids
        ref_grid = self.segments_to_multilabel_grid(segments, self.reference_set)
        hyp_grid = self.segments_to_multilabel_grid(segments, self.hypothesis_set)

        # Generate and save confusion matrices
        self.generate_confusion_matrices(ref_grid, hyp_grid, output_directory)

        # Compute and save classification report
        self.compute_classification_report(ref_grid, hyp_grid, output_directory)

        print(
            f"The reliability analysis results have been saved in {output_directory}."
        )


class ReliabilityPipeline(Pipeline):
    """
    Pipeline wrapper for the ReliabilityAnalysis class.
    """

    def __init__(self):
        self.project = None

    def run(
        self,
        path,
        destination,
        reference_set,
        hypothesis_set,
        speakers=None,
        granularity=10,
        func=None,
    ):
        """
        Execute the pipeline.
        """
        self.project = ChildProject(path)
        self.project.read()
        if destination is None:
            destination = os.path.join(
                path,
                "extra",
                f'reliability-{reference_set.replace("/","_")}-{hypothesis_set.replace("/","_")}',
            )

        analysis = ReliabilityAnalysis(
            self.project,
            reference_set=reference_set,
            hypothesis_set=hypothesis_set,
            speakers=speakers.split(",") if speakers else None,
            granularity=granularity,
        )
        analysis.extract(destination)

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help="Path to the dataset.")
        parser.add_argument(
            "--destination",
            default=None,
            help="Path to save the results. Defaults to {dataset}/extra/reliability-{reference-set}-{hypothesis-set}",
        )
        parser.add_argument(
            "--reference-set", "-r", required=True, help="Reference annotation set."
        )
        parser.add_argument(
            "--hypothesis-set", "-y", required=True, help="Hypothesis annotation set."
        )
        parser.add_argument(
            "--speakers",
            default="CHI,OCH,FEM,MAL",
            help="Comma-separated list of speakers.",
        )
        parser.add_argument(
            "--granularity",
            type=int,
            default=10,
            help="Grid granularity in milliseconds.",
        )
