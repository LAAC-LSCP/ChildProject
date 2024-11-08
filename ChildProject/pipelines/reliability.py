import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_grid, conf_matrix, segments_to_annotation
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

class ReliabilityAnalysis:
    def __init__(self, project: ChildProject, reference_set: str, hypothesis_set: str, speakers=None):
        """
        Initializes the reliability analysis.

        Args:
            project (ChildProject): The ChildProject object.
            reference_set (str): Name of the reference annotation set.
            hypothesis_set (str): Name of the hypothesis annotation set.
            speakers (list, optional): List of speakers to include in the analysis. Defaults to ['CHI', 'OCH', 'FEM', 'MAL'].
        """
        self.project = project
        self.reference_set = reference_set
        self.hypothesis_set = hypothesis_set
        self.speakers = speakers if speakers is not None else ['CHI', 'OCH', 'FEM', 'MAL']
        self.labels = self.speakers + ["Silence"]

    def load_data(self):
        """
        Loads the annotations and prepares them for comparison.
        """
        am = AnnotationManager(self.project)
        intersection = AnnotationManager.intersection(am.annotations, [self.reference_set, self.hypothesis_set])
        segments = am.get_collapsed_segments(intersection)
        segments = segments[segments['speaker_type'].isin(self.speakers)]
        return segments

    def generate_confusion_matrices(self, segments, output_directory):
        """
        Generates and saves confusion matrices.

        Args:
            segments (pd.DataFrame): Segments data.
            output_directory (str): Directory to save the confusion matrices.
        """
        # Create grids for reference and hypothesis sets
        grids = {}
        for annotation_set in [self.reference_set, self.hypothesis_set]:
            grids[annotation_set] = segments_to_grid(
                segments[segments['set'] == annotation_set],
                0,
                segments['segment_offset'].max(),
                100,
                'speaker_type',
                self.speakers
            )

        # Compute non-normalized confusion matrix
        confusion_counts = conf_matrix(grids[self.reference_set], grids[self.hypothesis_set])
        pd.DataFrame(confusion_counts, index=self.labels, columns=self.labels).to_csv(
            os.path.join(output_directory, "confusion_counts.csv"))

        # Plot and save non-normalized confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_counts, annot=True, cmap="Blues", fmt='d', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel("Hypothesis")
        plt.ylabel("Reference")
        plt.title("Non-Normalized Confusion Matrix")
        plt.savefig(os.path.join(output_directory, "confusion_counts.jpg"))
        plt.close()

        # Compute normalized confusion matrix
        confusion_normalized = confusion_counts / np.sum(grids[self.reference_set], axis=0)[:, None]
        pd.DataFrame(confusion_normalized, index=self.labels, columns=self.labels).to_csv(
            os.path.join(output_directory, "confusion_normalized.csv"))

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_normalized, annot=True, cmap="Reds", fmt='.2f', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel("Hypothesis")
        plt.ylabel("Reference")
        plt.title("Normalized Confusion Matrix")
        plt.savefig(os.path.join(output_directory, "confusion_normalized.jpg"))
        plt.close()

    def compute_detection_metrics(self, segments, output_directory):
        """
        Computes and saves detection metrics (Precision, Recall, F-measure).

        Args:
            segments (pd.DataFrame): Segments data.
            output_directory (str): Directory to save detection metrics.
        """
        ref = segments_to_annotation(segments[segments['set'] == self.reference_set], 'speaker_type')
        hyp = segments_to_annotation(segments[segments['set'] == self.hypothesis_set], 'speaker_type')

        metric = DetectionPrecisionRecallFMeasure()
        detail = metric.compute_components(ref, hyp)
        precision, recall, f_measure = metric.compute_metrics(detail)

        with open(os.path.join(output_directory, "detection_metrics.txt"), 'w') as f:
            f.write(f"Detection Metrics for {self.reference_set} vs {self.hypothesis_set}\n")
            f.write(f'Precision: {precision:.2f}\n')
            f.write(f'Recall: {recall:.2f}\n')
            f.write(f'F-measure: {f_measure:.2f}\n')

    def run(self, output_directory: str):
        """
        Runs the reliability analysis and saves the results.

        Args:
            output_directory (str): Path to the directory where results will be saved.
        """
        os.makedirs(output_directory, exist_ok=True)

        # Load data
        segments = self.load_data()

        # Generate confusion matrices
        self.generate_confusion_matrices(segments, output_directory)

        # Compute and save detection metrics
        self.compute_detection_metrics(segments, output_directory)

        print(f"The reliability analysis results have been saved in {output_directory}.")
