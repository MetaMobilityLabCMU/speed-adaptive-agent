"""
This module defines the SpeedCore class, which extends the Core class from mushroom_rl
to support learning and evaluation with speed adaptation.
"""
from collections import defaultdict
from mushroom_rl.core import Core
from tqdm import tqdm

class SpeedCore(Core):
    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, record=False, target_speed=1.25):
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        assert (render and record) or (not record), "To record, the render flag must be set to true"

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition = lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter >= self._n_episodes_per_fit

        self._run(
            n_steps,
            n_episodes,
            fit_condition,
            render,
            quiet,
            record,
            get_env_info=False,
            target_speed=target_speed)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, record=False, get_env_info=False, target_speed=1.25):
        assert (render and record) or (not record), "To record, the render flag must be set to true"

        fit_condition = lambda: False

        return self._run(
            n_steps,
            n_episodes,
            fit_condition,
            render,
            quiet,
            record,
            get_env_info,
            initial_states,
            target_speed=target_speed)

    def _run(
            self,
            n_steps,
            n_episodes,
            fit_condition,
            render,
            quiet,
            record,
            get_env_info,
            initial_states=None,
            target_speed=1.25):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition = lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps, dynamic_ncols=True, disable=quiet, leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(
                total=self._n_episodes,
                dynamic_ncols=True,
                disable=quiet,
                leave=False)

        dataset, dataset_info = self._run_impl(
            move_condition,
            fit_condition,
            steps_progress_bar,
            episodes_progress_bar,
            render,
            record,
            initial_states,
            target_speed=target_speed)

        if get_env_info:
            return dataset, dataset_info
        return dataset

    def _run_impl(
            self,
            move_condition,
            fit_condition,
            steps_progress_bar,
            episodes_progress_bar,
            render,
            record,
            initial_states,
            target_speed=1.25):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = []
        dataset_info = defaultdict(list)

        last = True
        while move_condition():
            if last:
                self.reset(initial_states)

            sample, step_info = self._step(render, record)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)

            for key, value in step_info.items():
                dataset_info[key].append(value)

            if fit_condition():
                # pull the corresponding expert demonstration
                dataset_info['target_speed'] = target_speed
                self.agent.fit(dataset, **dataset_info)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = []
                dataset_info = defaultdict(list)

            last = sample[-1]

        self.agent.stop()
        self.mdp.stop()

        if record:
            self._record.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset, dataset_info
    