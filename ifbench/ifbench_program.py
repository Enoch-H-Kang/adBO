# ifbench_program.py
"""
IFBench 2-Stage Program for GEPA Optimization.

From the GEPA paper:
"We design a 2-stage system, that first attempts to answer the user query,
and then in the second stage, rewrites the answer following the constraints."

Stage 1: Generate initial answer to the user query
Stage 2: Rewrite the answer to satisfy all output constraints
"""
from __future__ import annotations

from typing import List, Optional

import dspy


class InitialAnswerSig(dspy.Signature):
    """Generate an initial answer to the user's query."""
    prompt: str = dspy.InputField(desc="The user's instruction/query with embedded constraints.")
    answer: str = dspy.OutputField(desc="Initial answer addressing the user's query.")


class RewriteWithConstraintsSig(dspy.Signature):
    """Rewrite the answer to satisfy all output constraints."""
    prompt: str = dspy.InputField(desc="The original user instruction/query.")
    constraint_text: str = dspy.InputField(desc="The output constraints that must be satisfied.")
    initial_answer: str = dspy.InputField(desc="The initial answer to be rewritten.")
    final_answer: str = dspy.OutputField(desc="The rewritten answer satisfying all constraints.")


class ConstraintAwareAnswerSig(dspy.Signature):
    """Generate an answer while being mindful of output constraints."""
    prompt: str = dspy.InputField(desc="The user's instruction/query.")
    constraint_text: str = dspy.InputField(desc="Output constraints to satisfy (e.g., 'answer only yes/no', 'mention word 3 times').")
    answer: str = dspy.OutputField(desc="Answer that follows both the query requirements and output constraints.")


class IFBenchTwoStage(dspy.Module):
    """
    Two-stage instruction following program for IFBench.

    Stage 1: Generate initial answer (focus on content correctness)
    Stage 2: Rewrite to satisfy constraints (focus on format/constraint compliance)

    This mirrors the GEPA paper's approach where the system first attempts
    to answer the query, then rewrites following constraints.
    """

    def __init__(self, use_chain_of_thought: bool = True):
        super().__init__()

        if use_chain_of_thought:
            self.generate_initial = dspy.ChainOfThought(InitialAnswerSig)
            self.rewrite_with_constraints = dspy.ChainOfThought(RewriteWithConstraintsSig)
        else:
            self.generate_initial = dspy.Predict(InitialAnswerSig)
            self.rewrite_with_constraints = dspy.Predict(RewriteWithConstraintsSig)

    def forward(self, prompt: str, constraint_text: str = ""):
        """
        Two-stage forward pass.

        Args:
            prompt: The user's instruction/query (may have constraints embedded)
            constraint_text: Explicit constraint description (for feedback generation)

        Returns:
            dspy.Prediction with:
            - answer: Final answer satisfying constraints
            - initial_answer: First-stage answer
            - final_answer: Second-stage rewritten answer
        """
        # Stage 1: Generate initial answer
        stage1_result = self.generate_initial(prompt=prompt)
        initial_answer = stage1_result.answer

        # Stage 2: Rewrite with constraints
        # If no explicit constraint_text, try to extract from prompt or use generic guidance
        effective_constraint = constraint_text if constraint_text else self._extract_constraints_from_prompt(prompt)

        stage2_result = self.rewrite_with_constraints(
            prompt=prompt,
            constraint_text=effective_constraint,
            initial_answer=initial_answer,
        )
        final_answer = stage2_result.final_answer

        return dspy.Prediction(
            answer=final_answer,
            initial_answer=initial_answer,
            final_answer=final_answer,
            stage1_reasoning=getattr(stage1_result, "reasoning", ""),
            stage2_reasoning=getattr(stage2_result, "reasoning", ""),
        )

    def _extract_constraints_from_prompt(self, prompt: str) -> str:
        """
        Extract constraint-related parts from the prompt.

        Many IFBench prompts have constraints embedded at the end or as
        separate sentences. This is a simple heuristic extraction.
        """
        # Common constraint indicators
        indicators = [
            "must ", "should ", "ensure ", "make sure ",
            "at least ", "at most ", "exactly ",
            "do not ", "don't ", "never ",
            "only ", "always ",
            "format ", "output ",
            "word", "sentence", "paragraph",
            "lowercase", "uppercase", "capital",
            "json", "bullet", "number",
        ]

        lines = prompt.split("\n")
        constraint_lines = []

        for line in lines:
            line_lower = line.lower()
            if any(ind in line_lower for ind in indicators):
                constraint_lines.append(line.strip())

        if constraint_lines:
            return " ".join(constraint_lines)

        # Fallback: return last part of prompt which often contains constraints
        if len(prompt) > 100:
            return prompt[-200:]
        return prompt


class IFBenchSingleStage(dspy.Module):
    """
    Single-stage constraint-aware answering (simpler baseline).

    This is an alternative to the two-stage approach where the model
    directly generates an answer while considering constraints.
    """

    def __init__(self, use_chain_of_thought: bool = True):
        super().__init__()

        if use_chain_of_thought:
            self.generate_answer = dspy.ChainOfThought(ConstraintAwareAnswerSig)
        else:
            self.generate_answer = dspy.Predict(ConstraintAwareAnswerSig)

    def forward(self, prompt: str, constraint_text: str = ""):
        """
        Single-stage forward pass.

        Args:
            prompt: The user's instruction/query
            constraint_text: Constraint description

        Returns:
            dspy.Prediction with answer
        """
        result = self.generate_answer(
            prompt=prompt,
            constraint_text=constraint_text if constraint_text else "Follow all instructions in the prompt.",
        )

        return dspy.Prediction(
            answer=result.answer,
            initial_answer=result.answer,  # Same as final for single stage
            final_answer=result.answer,
            reasoning=getattr(result, "reasoning", ""),
        )


class IFBenchIterativeRefine(dspy.Module):
    """
    Iterative refinement approach for instruction following.

    This variant allows multiple refinement iterations based on
    constraint feedback (useful for complex multi-constraint scenarios).
    """

    def __init__(self, max_refinements: int = 2, use_chain_of_thought: bool = True):
        super().__init__()
        self.max_refinements = max_refinements

        if use_chain_of_thought:
            self.generate_initial = dspy.ChainOfThought(InitialAnswerSig)
            self.refine = dspy.ChainOfThought(RewriteWithConstraintsSig)
        else:
            self.generate_initial = dspy.Predict(InitialAnswerSig)
            self.refine = dspy.Predict(RewriteWithConstraintsSig)

    def forward(
        self,
        prompt: str,
        constraint_text: str = "",
        constraint_feedback: Optional[str] = None,
    ):
        """
        Iterative refinement forward pass.

        Args:
            prompt: The user's instruction/query
            constraint_text: Constraint description
            constraint_feedback: Optional feedback about which constraints failed

        Returns:
            dspy.Prediction with final and intermediate answers
        """
        # Stage 1: Initial answer
        stage1_result = self.generate_initial(prompt=prompt)
        current_answer = stage1_result.answer
        initial_answer = current_answer

        effective_constraint = constraint_text if constraint_text else prompt

        # If feedback is provided, incorporate it into constraints
        if constraint_feedback:
            effective_constraint = f"{effective_constraint}\n\nFeedback on previous attempt:\n{constraint_feedback}"

        # Refinement iterations
        refinement_history = [initial_answer]

        for i in range(self.max_refinements):
            refine_result = self.refine(
                prompt=prompt,
                constraint_text=effective_constraint,
                initial_answer=current_answer,
            )
            current_answer = refine_result.final_answer
            refinement_history.append(current_answer)

        return dspy.Prediction(
            answer=current_answer,
            initial_answer=initial_answer,
            final_answer=current_answer,
            refinement_history=refinement_history,
        )


# Factory function to create the appropriate program
def create_ifbench_program(
    variant: str = "two_stage",
    use_chain_of_thought: bool = True,
    max_refinements: int = 2,
) -> dspy.Module:
    """
    Factory function to create IFBench programs.

    Args:
        variant: One of "two_stage", "single_stage", "iterative"
        use_chain_of_thought: Whether to use CoT reasoning
        max_refinements: Number of refinement iterations (for iterative variant)

    Returns:
        Configured dspy.Module
    """
    if variant == "two_stage":
        return IFBenchTwoStage(use_chain_of_thought=use_chain_of_thought)
    elif variant == "single_stage":
        return IFBenchSingleStage(use_chain_of_thought=use_chain_of_thought)
    elif variant == "iterative":
        return IFBenchIterativeRefine(
            max_refinements=max_refinements,
            use_chain_of_thought=use_chain_of_thought,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'two_stage', 'single_stage', or 'iterative'")


if __name__ == "__main__":
    # Quick test
    import dspy

    # Mock LM for testing
    class MockLM:
        def __call__(self, *args, **kwargs):
            return "Mock response"

    # Test two-stage program structure
    program = IFBenchTwoStage(use_chain_of_thought=False)
    print("IFBenchTwoStage modules:")
    for name, module in program.named_predictors():
        print(f"  {name}: {module}")

    print("\nProgram signatures:")
    print(f"  Stage 1 (InitialAnswerSig): prompt -> answer")
    print(f"  Stage 2 (RewriteWithConstraintsSig): prompt, constraint_text, initial_answer -> final_answer")
