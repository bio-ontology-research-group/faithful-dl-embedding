/-
  GeometricFaithful/Balls.lean
  ============================
  Balls of positive radius: if the intersection of two closed balls is a closed
  ball of positive radius (in a real inner product space of dimension ≥ 2), the
  two balls are nested.  Hence the equivalences of `T₃` force the three `Xᵢ`
  balls to be pairwise nested, so one of them is contained in the other two and
  `A ⊓ B` is non-empty — no ball model of `T₃` has `A`, `B`, `C` all non-empty.
-/
import GeometricFaithful.Theory

namespace GeometricFaithful

open Metric ContextCalculus.ELCompletion

-- `Type` rather than `Type*`: the kobayashi-marust interpretations live in universe 0.
variable {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The space contains three distinct unit vectors (true as soon as `dim ≥ 2`). -/
def ThreeUnitVectors (E : Type) [NormedAddCommGroup E] : Prop :=
  ∃ e₀ e₁ e₂ : E, ‖e₀‖ = 1 ∧ ‖e₁‖ = 1 ∧ ‖e₂‖ = 1 ∧ e₀ ≠ e₁ ∧ e₀ ≠ e₂ ∧ e₁ ≠ e₂

/-- If `closedBall a ρ ⊆ closedBall c r` then `‖a - c‖ + ρ ≤ r` (in a space with a
unit vector, so that balls of different radii differ). -/
theorem norm_sub_add_le_of_closedBall_subset (hE : ∃ e : E, ‖e‖ = 1) {a c : E} {ρ r : ℝ}
    (hρ : 0 ≤ ρ) (hsub : closedBall a ρ ⊆ closedBall c r) : ‖a - c‖ + ρ ≤ r := by
  by_cases hac : a = c
  · subst hac
    obtain ⟨e, he⟩ := hE
    have hz : a + ρ • e ∈ closedBall a ρ := by
      rw [mem_closedBall, dist_eq_norm, add_sub_cancel_left, norm_smul, he, mul_one,
        Real.norm_of_nonneg hρ]
    have := hsub hz
    rw [mem_closedBall, dist_eq_norm, add_sub_cancel_left, norm_smul, he, mul_one,
      Real.norm_of_nonneg hρ] at this
    simpa using this
  · set u : E := ‖a - c‖⁻¹ • (a - c) with hu
    have hnorm : ‖a - c‖ ≠ 0 := norm_ne_zero_iff.2 (sub_ne_zero.2 hac)
    have hu1 : ‖u‖ = 1 := by
      rw [hu, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hnorm]
    have hz : a + ρ • u ∈ closedBall a ρ := by
      rw [mem_closedBall, dist_eq_norm, add_sub_cancel_left, norm_smul, hu1, mul_one,
        Real.norm_of_nonneg hρ]
    have := hsub hz
    rw [mem_closedBall, dist_eq_norm] at this
    have hcalc : a + ρ • u - c = (1 + ρ * ‖a - c‖⁻¹) • (a - c) := by
      rw [hu, smul_smul, add_smul, one_smul]; abel
    rw [hcalc, norm_smul, Real.norm_of_nonneg (by positivity), add_mul, one_mul,
      mul_assoc, inv_mul_cancel₀ hnorm, mul_one] at this
    linarith

/-- Points of the small sphere lie on one of the two big spheres. -/
theorem sphere_point_on_boundary {c₁ c₂ a : E} {r₁ r₂ ρ : ℝ}
    (hK : closedBall c₁ r₁ ∩ closedBall c₂ r₂ = closedBall a ρ)
    {v : E} (hv : ‖v‖ = 1) (hρ : 0 ≤ ρ) :
    dist (a + ρ • v) c₁ = r₁ ∨ dist (a + ρ • v) c₂ = r₂ := by
  set x := a + ρ • v with hx
  have hxK : x ∈ closedBall a ρ := by
    rw [mem_closedBall, dist_eq_norm, hx, add_sub_cancel_left, norm_smul, hv, mul_one,
      Real.norm_of_nonneg hρ]
  rw [← hK] at hxK
  obtain ⟨hx₁, hx₂⟩ := hxK
  rw [mem_closedBall] at hx₁ hx₂
  by_contra hcon
  obtain ⟨hne₁, hne₂⟩ := not_or.1 hcon
  have hlt₁ : dist x c₁ < r₁ := lt_of_le_of_ne hx₁ hne₁
  have hlt₂ : dist x c₂ < r₂ := lt_of_le_of_ne hx₂ hne₂
  set ε := min (r₁ - dist x c₁) (r₂ - dist x c₂) with hε
  have hεpos : 0 < ε := lt_min (by linarith) (by linarith)
  have hε₁ : ε ≤ r₁ - dist x c₁ := min_le_left _ _
  have hε₂ : ε ≤ r₂ - dist x c₂ := min_le_right _ _
  set y := a + (ρ + ε) • v with hy
  have hyx : dist y x = ε := by
    rw [dist_eq_norm, hy, hx, add_sub_add_left_eq_sub, ← sub_smul, add_sub_cancel_left,
      norm_smul, hv, mul_one, Real.norm_of_nonneg hεpos.le]
  have hy₁ : y ∈ closedBall c₁ r₁ := by
    rw [mem_closedBall]
    calc dist y c₁ ≤ dist y x + dist x c₁ := dist_triangle _ _ _
      _ ≤ r₁ := by rw [hyx]; linarith
  have hy₂ : y ∈ closedBall c₂ r₂ := by
    rw [mem_closedBall]
    calc dist y c₂ ≤ dist y x + dist x c₂ := dist_triangle _ _ _
      _ ≤ r₂ := by rw [hyx]; linarith
  have hyK : y ∈ closedBall a ρ := hK ▸ ⟨hy₁, hy₂⟩
  rw [mem_closedBall, dist_eq_norm, hy, add_sub_cancel_left, norm_smul, hv, mul_one,
    Real.norm_of_nonneg (by linarith)] at hyK
  linarith

/-- On a small sphere strictly inside a big ball, at most one point lies on the
big sphere: equality in the triangle inequality forces the direction. -/
theorem unit_vector_unique_of_on_sphere {c a : E} {r ρ : ℝ} (hρ : 0 < ρ)
    (hsub : closedBall a ρ ⊆ closedBall c r) (hne : closedBall a ρ ≠ closedBall c r)
    {v w : E} (hv : ‖v‖ = 1) (hw : ‖w‖ = 1)
    (hv' : dist (a + ρ • v) c = r) (hw' : dist (a + ρ • w) c = r) : v = w := by
  have hbound := norm_sub_add_le_of_closedBall_subset ⟨v, hv⟩ hρ.le hsub
  -- a unit vector `u` with `dist (a + ρ • u) c = r` equals `‖a - c‖⁻¹ • (a - c)`
  have key : ∀ u : E, ‖u‖ = 1 → dist (a + ρ • u) c = r → a ≠ c ∧ u = ‖a - c‖⁻¹ • (a - c) := by
    intro u hu hu'
    have h1 : ‖ρ • u + (a - c)‖ = ‖ρ • u‖ + ‖a - c‖ := by
      have hρu : ‖ρ • u‖ = ρ := by rw [norm_smul, hu, mul_one, Real.norm_of_nonneg hρ.le]
      rw [hρu]
      have : a + ρ • u - c = ρ • u + (a - c) := by abel
      rw [dist_eq_norm, this] at hu'
      have htri : ‖ρ • u + (a - c)‖ ≤ ‖ρ • u‖ + ‖a - c‖ := norm_add_le _ _
      rw [hρu] at htri
      linarith
    have hray : SameRay ℝ (ρ • u) (a - c) := (sameRay_iff_norm_add).2 h1
    rw [sameRay_iff_inv_norm_smul_eq] at hray
    have hρu0 : ρ • u ≠ 0 := by
      rw [smul_ne_zero_iff]; exact ⟨hρ.ne', norm_ne_zero_iff.1 (by rw [hu]; exact one_ne_zero)⟩
    rcases hray with h0 | h0 | heq
    · exact absurd h0 hρu0
    · -- a = c: then ρ = r and the balls coincide
      exfalso
      have hac : a = c := sub_eq_zero.1 h0
      subst hac
      have hr : r = ρ := by
        rw [dist_eq_norm, add_sub_cancel_left, norm_smul, hu, mul_one,
          Real.norm_of_nonneg hρ.le] at hu'
        exact hu'.symm
      exact hne (by rw [hr])
    · refine ⟨fun hac => ?_, ?_⟩
      · apply hρu0
        have : a - c = 0 := sub_eq_zero.2 hac
        rw [this, norm_zero, inv_zero, zero_smul] at heq
        have hρu : ‖ρ • u‖ = ρ := by rw [norm_smul, hu, mul_one, Real.norm_of_nonneg hρ.le]
        rw [hρu] at heq
        have := congrArg (fun z => ρ • z) heq
        simpa [smul_smul, mul_inv_cancel₀ hρ.ne'] using this
      · have hρu : ‖ρ • u‖ = ρ := by rw [norm_smul, hu, mul_one, Real.norm_of_nonneg hρ.le]
        rw [hρu, smul_smul, inv_mul_cancel₀ hρ.ne', one_smul] at heq
        exact heq
  obtain ⟨_, hv''⟩ := key v hv hv'
  obtain ⟨_, hw''⟩ := key w hw hw'
  rw [hv'', hw'']

/-- **Two closed balls of positive radius whose intersection is a closed ball of
positive radius are nested** (in a space with three distinct unit vectors). -/
theorem nested_of_inter_eq_closedBall (h3 : ThreeUnitVectors E)
    {c₁ c₂ a : E} {r₁ r₂ ρ : ℝ} (hρ : 0 < ρ)
    (hK : closedBall c₁ r₁ ∩ closedBall c₂ r₂ = closedBall a ρ) :
    closedBall c₁ r₁ ⊆ closedBall c₂ r₂ ∨ closedBall c₂ r₂ ⊆ closedBall c₁ r₁ := by
  by_contra hcon
  obtain ⟨hn₁₂, hn₂₁⟩ := not_or.1 hcon
  have hsub₁ : closedBall a ρ ⊆ closedBall c₁ r₁ := hK ▸ Set.inter_subset_left
  have hsub₂ : closedBall a ρ ⊆ closedBall c₂ r₂ := hK ▸ Set.inter_subset_right
  have hne₁ : closedBall a ρ ≠ closedBall c₁ r₁ := fun heq => hn₁₂ (heq ▸ hsub₂)
  have hne₂ : closedBall a ρ ≠ closedBall c₂ r₂ := fun heq => hn₂₁ (heq ▸ hsub₁)
  obtain ⟨e₀, e₁, e₂, h₀, h₁, h₂, h₀₁, h₀₂, h₁₂⟩ := h3
  have b₀ := sphere_point_on_boundary hK h₀ hρ.le
  have b₁ := sphere_point_on_boundary hK h₁ hρ.le
  have b₂ := sphere_point_on_boundary hK h₂ hρ.le
  have u₁ : ∀ {v w : E}, ‖v‖ = 1 → ‖w‖ = 1 → dist (a + ρ • v) c₁ = r₁ →
      dist (a + ρ • w) c₁ = r₁ → v = w :=
    fun hv hw => unit_vector_unique_of_on_sphere hρ hsub₁ hne₁ hv hw
  have u₂ : ∀ {v w : E}, ‖v‖ = 1 → ‖w‖ = 1 → dist (a + ρ • v) c₂ = r₂ →
      dist (a + ρ • w) c₂ = r₂ → v = w :=
    fun hv hw => unit_vector_unique_of_on_sphere hρ hsub₂ hne₂ hv hw
  rcases b₀ with b₀ | b₀ <;> rcases b₁ with b₁ | b₁ <;> rcases b₂ with b₂ | b₂
  · exact h₀₁ (u₁ h₀ h₁ b₀ b₁)
  · exact h₀₁ (u₁ h₀ h₁ b₀ b₁)
  · exact h₀₂ (u₁ h₀ h₂ b₀ b₂)
  · exact h₁₂ (u₂ h₁ h₂ b₁ b₂)
  · exact h₁₂ (u₁ h₁ h₂ b₁ b₂)
  · exact h₀₂ (u₂ h₀ h₂ b₀ b₂)
  · exact h₀₁ (u₂ h₀ h₁ b₀ b₁)
  · exact h₀₁ (u₂ h₀ h₁ b₀ b₁)

/-- **No ball model of `T₃`.** If all six names are interpreted by closed balls of
positive radius (in a real inner product space of dimension ≥ 2), the model is
degenerate. -/
theorem no_ball_model (h3 : ThreeUnitVectors E) (I : GInterp E) (h : models I T3)
    {c₁ c₂ c₃ cA cB cC : E} {r₁ r₂ r₃ rA rB rC : ℝ}
    (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₃ : 0 < r₃)
    (hrA : 0 < rA) (hrB : 0 < rB) (hrC : 0 < rC)
    (h₁ : region I .X1 = closedBall c₁ r₁) (h₂ : region I .X2 = closedBall c₂ r₂)
    (h₃ : region I .X3 = closedBall c₃ r₃)
    (hA : region I .A = closedBall cA rA) (hB : region I .B = closedBall cB rB)
    (hC : region I .C = closedBall cC rC) : False := by
  have eA := region_A h
  have eB := region_B h
  have eC := region_C h
  rw [hA, h₁, h₂] at eA
  rw [hB, h₂, h₃] at eB
  rw [hC, h₁, h₃] at eC
  have n₁₂ := nested_of_inter_eq_closedBall h3 hrA eA.symm
  have n₂₃ := nested_of_inter_eq_closedBall h3 hrB eB.symm
  have n₁₃ := nested_of_inter_eq_closedBall h3 hrC eC.symm
  have hempty := triple_inter_empty h
  rw [h₁, h₂, h₃] at hempty
  have m₁ : c₁ ∈ closedBall c₁ r₁ := mem_closedBall_self hr₁.le
  have m₂ : c₂ ∈ closedBall c₂ r₂ := mem_closedBall_self hr₂.le
  have m₃ : c₃ ∈ closedBall c₃ r₃ := mem_closedBall_self hr₃.le
  have : (closedBall c₁ r₁ ∩ closedBall c₂ r₂ ∩ closedBall c₃ r₃).Nonempty := by
    rcases n₁₂ with s₁₂ | s₂₁ <;> rcases n₂₃ with s₂₃ | s₃₂ <;> rcases n₁₃ with s₁₃ | s₃₁
    · exact ⟨c₁, ⟨m₁, s₁₂ m₁⟩, s₁₃ m₁⟩
    · exact ⟨c₁, ⟨m₁, s₁₂ m₁⟩, s₂₃ (s₁₂ m₁)⟩
    · exact ⟨c₁, ⟨m₁, s₁₂ m₁⟩, s₁₃ m₁⟩
    · exact ⟨c₃, ⟨s₃₁ m₃, s₃₂ m₃⟩, m₃⟩
    · exact ⟨c₂, ⟨s₂₁ m₂, m₂⟩, s₂₃ m₂⟩
    · exact ⟨c₂, ⟨s₂₁ m₂, m₂⟩, s₂₃ m₂⟩
    · exact ⟨c₃, ⟨s₂₁ (s₃₂ m₃), s₃₂ m₃⟩, m₃⟩
    · exact ⟨c₃, ⟨s₃₁ m₃, s₃₂ m₃⟩, m₃⟩
  rw [hempty] at this
  exact Set.not_nonempty_empty this

/-- Euclidean space of dimension ≥ 2 has three distinct unit vectors. -/
theorem threeUnitVectors_euclideanSpace {d : ℕ} (hd : 2 ≤ d) :
    ThreeUnitVectors (EuclideanSpace ℝ (Fin d)) := by
  refine ⟨EuclideanSpace.single ⟨0, by omega⟩ 1, -EuclideanSpace.single ⟨0, by omega⟩ 1,
    EuclideanSpace.single ⟨1, by omega⟩ 1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp
  · simp
  · intro heq
    have := congrArg (fun v : EuclideanSpace ℝ (Fin d) => v ⟨0, by omega⟩) heq
    norm_num [PiLp.single_apply] at this
  · intro heq
    have := congrArg (fun v : EuclideanSpace ℝ (Fin d) => v ⟨0, by omega⟩) heq
    norm_num [PiLp.single_apply] at this
  · intro heq
    have := congrArg (fun v : EuclideanSpace ℝ (Fin d) => v ⟨0, by omega⟩) heq
    norm_num [PiLp.single_apply] at this

/-- **No ball model of `T₃` in `ℝᵈ`, `d ≥ 2`.** -/
theorem no_ball_model_euclidean {d : ℕ} (hd : 2 ≤ d)
    (I : GInterp (EuclideanSpace ℝ (Fin d))) (h : models I T3)
    {c₁ c₂ c₃ cA cB cC : EuclideanSpace ℝ (Fin d)} {r₁ r₂ r₃ rA rB rC : ℝ}
    (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₃ : 0 < r₃)
    (hrA : 0 < rA) (hrB : 0 < rB) (hrC : 0 < rC)
    (h₁ : region I .X1 = closedBall c₁ r₁) (h₂ : region I .X2 = closedBall c₂ r₂)
    (h₃ : region I .X3 = closedBall c₃ r₃)
    (hA : region I .A = closedBall cA rA) (hB : region I .B = closedBall cB rB)
    (hC : region I .C = closedBall cC rC) : False :=
  no_ball_model (threeUnitVectors_euclideanSpace hd) I h hr₁ hr₂ hr₃ hrA hrB hrC h₁ h₂ h₃ hA hB hC

end GeometricFaithful
