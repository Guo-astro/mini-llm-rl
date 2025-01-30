import re


class RewardFunctions:
    @staticmethod
    def format_reward(completions, targets):
        """
        Validates correct answer format using <think>...</think><answer>...</answer>.
        """
        rewards = []
        for completion, target in zip(completions, targets):
            try:
                completion = "<think>" + completion  # Ensure <think> is always included
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                match = re.search(regex, completion, re.DOTALL)
                rewards.append(1.0 if match else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def equation_reward(completions, targets, numbers_list):
        """
        Evaluates completion accuracy using mathematical correctness and format validity.
        """
        rewards = []
        for completion, target, numbers in zip(completions, targets, numbers_list):
            try:
                completion = "<think>" + completion  # Ensure <think> is always included
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if not match:
                    rewards.append(0.0)
                    continue

                equation = match.group(1).strip()
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue

                # Validate equation syntax
                if not re.match(r'^[\d+\-*/().\s]+$', equation):
                    rewards.append(0.0)
                    continue

                result = eval(equation, {"__builtins__": None}, {})
                rewards.append(1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards
