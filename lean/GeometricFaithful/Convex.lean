/-
  GeometricFaithful/Convex.lean
  =============================
  Convexity and connectedness are *not* the obstruction:

  * `T₃` has a model by convex regions with non-empty interior in the plane
    (`planeModel`), in which `A`, `B`, `C` are non-empty.
  * The correct statement for convex regions is a dimension bound, via Helly's
    theorem: the `k`-concept theory `T_k` (every `k-1` of `X₁..X_k` jointly
    non-empty, all `k` disjoint) has no non-degenerate convex model in a space of
    dimension `≤ k-2`.
-/
import GeometricFaithful.Theory

namespace GeometricFaithful

open ContextCalculus.ELCompletion

/-! ### A convex model of `T₃` in the plane -/

/-- Half-planes `x₀ ≤ 1/4`, `x₁ ≤ 1/4`, `x₀ + x₁ ≥ 3/4` in `ℝ²`. -/
def planeModel : GInterp (Fin 2 → ℝ) where
  concept
    | .X1 => fun x => x 0 ≤ 1/4
    | .X2 => fun x => x 1 ≤ 1/4
    | .X3 => fun x => 3/4 ≤ x 0 + x 1
    | .A => fun x => x 0 ≤ 1/4 ∧ x 1 ≤ 1/4
    | .B => fun x => x 1 ≤ 1/4 ∧ 3/4 ≤ x 0 + x 1
    | .C => fun x => x 0 ≤ 1/4 ∧ 3/4 ≤ x 0 + x 1
    | .top => fun _ => True
    | .bot => fun _ => False
  role := fun r => r.elim
  top_true := fun _ => trivial
  bottom_false := fun _ h => h

theorem planeModel_models : models planeModel T3 := by
  intro c hc
  simp only [T3, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [satClause, planeModel] <;> intro x <;> first
    | (intro h; exact h.1)
    | (intro h; exact h.2)
    | (intro h₁ h₂; exact ⟨h₁, h₂⟩)
    | (intro h₁ h₂; linarith [h₁.1, h₁.2, h₂.1, h₂.2])

theorem planeModel_nondegenerate :
    (region planeModel .A).Nonempty ∧ (region planeModel .B).Nonempty ∧
    (region planeModel .C).Nonempty := by
  refine ⟨⟨fun _ => 0, ?_⟩, ⟨![1, 0], ?_⟩, ⟨![0, 1], ?_⟩⟩
  · show (0 : ℝ) ≤ 1/4 ∧ (0 : ℝ) ≤ 1/4
    norm_num
  · show (![1, 0] : Fin 2 → ℝ) 1 ≤ 1/4 ∧ 3/4 ≤ (![1, 0] : Fin 2 → ℝ) 0 + (![1, 0] : Fin 2 → ℝ) 1
    simp; norm_num
  · show (![0, 1] : Fin 2 → ℝ) 0 ≤ 1/4 ∧ 3/4 ≤ (![0, 1] : Fin 2 → ℝ) 0 + (![0, 1] : Fin 2 → ℝ) 1
    simp; norm_num

theorem planeModel_convex :
    Convex ℝ (region planeModel .X1) ∧ Convex ℝ (region planeModel .X2) ∧
    Convex ℝ (region planeModel .X3) := by
  refine ⟨?_, ?_, ?_⟩
  · intro x hx y hy a b ha hb hab
    simp only [region, planeModel, Set.mem_setOf_eq, Pi.add_apply, Pi.smul_apply,
      smul_eq_mul] at hx hy ⊢
    nlinarith
  · intro x hx y hy a b ha hb hab
    simp only [region, planeModel, Set.mem_setOf_eq, Pi.add_apply, Pi.smul_apply,
      smul_eq_mul] at hx hy ⊢
    nlinarith
  · intro x hx y hy a b ha hb hab
    simp only [region, planeModel, Set.mem_setOf_eq, Pi.add_apply, Pi.smul_apply,
      smul_eq_mul] at hx hy ⊢
    nlinarith

/-! ### Helly's theorem: the dimension must grow with `k` -/

/-- **Helly dimension bound.** Let `X₁, …, X_k` be convex subsets of a real
vector space `E` of finite dimension `d ≤ k - 2`, and let `Aᵢ = ⋂_{j ≠ i} Xⱼ`.
If all `Aᵢ` are non-empty, two of them intersect (indeed all `Xⱼ` share a
point). So the theory `T_k` has no non-degenerate convex model in dimension
`≤ k - 2`. -/
theorem helly_dimension_bound {E : Type*} [AddCommGroup E] [Module ℝ E]
    [FiniteDimensional ℝ E] {k : ℕ} (hk : 2 ≤ k)
    (hd : Module.finrank ℝ E + 2 ≤ k)
    (X A : Fin k → Set E) (hconv : ∀ j, Convex ℝ (X j))
    (hA : ∀ i, A i = ⋂ j ∈ ({j | j ≠ i} : Set (Fin k)), X j)
    (hne : ∀ i, (A i).Nonempty) :
    ∃ i j, i ≠ j ∧ (A i ∩ A j).Nonempty := by
  classical
  -- every `d+1` of the `X j` intersect, because they are among some `k-1` of them
  have hinter : ∀ I : Finset (Fin k), I ⊆ Finset.univ →
      I.card = Module.finrank ℝ E + 1 → (⋂ j ∈ I, X j).Nonempty := by
    intro I _ hcard
    have hlt : I.card < (Finset.univ : Finset (Fin k)).card := by
      rw [Finset.card_univ, Fintype.card_fin]; omega
    obtain ⟨i, _, hi⟩ := Finset.exists_mem_notMem_of_card_lt_card hlt
    obtain ⟨p, hp⟩ := hne i
    refine ⟨p, ?_⟩
    rw [hA i] at hp
    simp only [Set.mem_iInter, Set.mem_setOf_eq] at hp
    simp only [Set.mem_iInter]
    intro j hj
    exact hp j (fun hji => hi (hji ▸ hj))
  have hall : (⋂ j ∈ (Finset.univ : Finset (Fin k)), X j).Nonempty :=
    Convex.helly_theorem (𝕜 := ℝ) (by rw [Finset.card_univ, Fintype.card_fin]; omega)
      (fun j _ => hconv j) hinter
  obtain ⟨p, hp⟩ := hall
  simp only [Set.mem_iInter, Finset.mem_univ, true_implies] at hp
  refine ⟨⟨0, by omega⟩, ⟨1, by omega⟩, by simp [Fin.ext_iff], ⟨p, ?_, ?_⟩⟩
  · rw [hA]; simp only [Set.mem_iInter, Set.mem_setOf_eq]; intro j _; exact hp j
  · rw [hA]; simp only [Set.mem_iInter, Set.mem_setOf_eq]; intro j _; exact hp j

end GeometricFaithful
