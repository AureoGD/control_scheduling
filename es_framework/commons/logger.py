import os
import csv
import time
from datetime import datetime
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from es_framework.commons.nn_parameters import unflatten_nn_parameters


class TrainingLogger:

    def __init__(
            self,
            alg: str = 'cmaes',  # 'cames' or 'cem'
            discrete: bool = False,
            log_to_csv: bool = True,
            log_to_tensorboard: bool = True,
            save_overall_best_model: bool = True,
            save_periodic_best_model: bool = True,
            periodic_best_model_interval: int = 50):

        self.algorithm_name = alg
        self.log_to_csv = log_to_csv
        self.log_to_tensorboard = log_to_tensorboard if SummaryWriter is not None else False
        self.save_overall_best_model_flag = save_overall_best_model
        self.save_periodic_best_model_flag = save_periodic_best_model
        self.periodic_best_model_interval = periodic_best_model_interval
        self.last_generation_end_time = time.time()

        if discrete:
            prefix = 'D_'
        else:
            prefix = 'C_'

        # Get the time string
        time_str = datetime.now().strftime("%d%H%M%S")

        # Combine them in the new order
        run_name = prefix + time_str

        self.tb_log_dir = os.path.join('logs', self.algorithm_name, run_name)

        self.tb_writer = None
        if self.log_to_tensorboard:
            try:
                os.makedirs(self.tb_log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=self.tb_log_dir)
                print(f"Logging to TensorBoard: {self.tb_log_dir}")
            except Exception as e:
                print(f"TensorBoard init error: {e}")
                self.log_to_tensorboard = False

        self.models_save_dir = os.path.join('models', self.algorithm_name, run_name)
        if self.save_overall_best_model_flag or self.save_periodic_best_model_flag:
            os.makedirs(self.models_save_dir, exist_ok=True)
            print(f"Models will be saved to: {self.models_save_dir}")

        self.csv_filepath = os.path.join(self.tb_log_dir, "training_log.csv")
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers_base = [
            "generation", "timestamp", "generation_duration_sec", "best_fitness_in_gen", "mean_fitness", "std_fitness",
            "min_fitness", "population_size", "overall_best_fitness"
        ]
        self.csv_headers = self.csv_headers_base.copy()

        if self.log_to_csv:
            try:
                self.csv_file = open(self.csv_filepath, 'w', newline='')
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_headers)
                self.csv_writer.writeheader()
                print(f"Logging to CSV: {self.csv_filepath}")
            except IOError as e:
                print(f"CSV file error: {e}")
                self.log_to_csv = False

        self.start_time = time.time()
        self.overall_best_fitness = -np.inf

    def set_reference_model(self, model: torch.nn.Module):
        self.reference_model = model

    def log_generation(self,
                       generation: int,
                       evaluated_population: List[Tuple[np.ndarray, float]],
                       extra_metrics: Optional[Dict[str, float]] = None):

        if not evaluated_population:
            print("Warning: Empty population.")
            return

        fitness_scores = [score for _, score in evaluated_population]
        best_fitness_in_gen = np.max(fitness_scores)
        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        min_fitness = np.min(fitness_scores)
        population_size = len(fitness_scores)

        timestamp_console = time.strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()
        generation_duration = current_time - self.last_generation_end_time
        self.last_generation_end_time = current_time

        best_params = evaluated_population[np.argmax(fitness_scores)][0]

        new_best = False
        if best_fitness_in_gen > self.overall_best_fitness:
            self.overall_best_fitness = best_fitness_in_gen
            new_best = True
            if self.save_overall_best_model_flag:
                self.save_model_checkpoint("overall_best_model.pth", best_params)

        if self.save_periodic_best_model_flag and generation % self.periodic_best_model_interval == 0:
            self.save_model_checkpoint(f"best_of_gen_{generation:04d}.pth", best_params)

        log_entry = {
            "generation": generation,
            "timestamp": timestamp_console,
            "generation_duration_sec": f"{generation_duration:.2f}",
            "best_fitness_in_gen": f"{best_fitness_in_gen:.4f}",
            "mean_fitness": f"{mean_fitness:.4f}",
            "std_fitness": f"{std_fitness:.4f}",
            "min_fitness": f"{min_fitness:.4f}",
            "population_size": population_size,
            "overall_best_fitness": f"{self.overall_best_fitness:.4f}"
        }

        if extra_metrics:
            for k, v in extra_metrics.items():
                log_entry[k] = f"{v:.6f}"
                if k not in self.csv_headers:
                    self.csv_headers.append(k)

        # Update CSV headers dynamically
        if generation == 0 and self.csv_writer:
            self.csv_writer.fieldnames = self.csv_headers
            self.csv_writer.writeheader()

        if self.log_to_csv and self.csv_writer:
            self.csv_writer.writerow(log_entry)
            self.csv_file.flush()

        if self.log_to_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar("Fitness/Best_in_Generation", best_fitness_in_gen, generation)
            self.tb_writer.add_scalar("Fitness/Mean", mean_fitness, generation)
            self.tb_writer.add_scalar("Fitness/StdDev", std_fitness, generation)
            self.tb_writer.add_scalar("Fitness/Min", min_fitness, generation)
            self.tb_writer.add_scalar("Fitness/Overall_Best", self.overall_best_fitness, generation)
            self.tb_writer.add_scalar("Population/Size", population_size, generation)
            self.tb_writer.add_scalar("Timing/Generation_Duration_sec", generation_duration, generation)
            if extra_metrics:
                for k, v in extra_metrics.items():
                    self.tb_writer.add_scalar(f"ExtraMetrics/{k}", v, generation)
            self.tb_writer.flush()

        print(
            f"Gen {generation:03d} | Time: {generation_duration:.2f}s | "
            f"Best: {best_fitness_in_gen:.4f} | Mean: {mean_fitness:.4f} | Std: {std_fitness:.4f} | New Best: {new_best}"
        )

    def save_model_checkpoint(self, filename: str, params_flat: np.ndarray):
        if not hasattr(self, 'reference_model') or self.reference_model is None:
            return
        try:
            state_dict = unflatten_nn_parameters(params_flat, self.reference_model)
            filepath = os.path.join(self.models_save_dir, filename)
            torch.save(state_dict, filepath)
            print(f"Saved model: {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def log_message(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        if self.log_to_tensorboard and self.tb_writer:
            self.tb_writer.add_text("Messages", f"[{timestamp}] {message}", global_step=int(time.time()))

    def close(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"CSV log closed: {self.csv_filepath}")
            except Exception as e:
                print(f"Error closing CSV: {e}")

        if self.tb_writer:
            try:
                self.tb_writer.close()
                print("TensorBoard writer closed.")
            except Exception as e:
                print(f"Error closing TensorBoard writer: {e}")

        total_time = time.time() - self.start_time
        self.log_message(f"Training finished. Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
