/-
  GeometricFaithful/Boxes.lean
  ============================
  Intervals (order-connected subsets of ℝ) and axis-parallel boxes (products of
  intervals) have Helly number 2: three pairwise intersecting members have a
  common point.  Hence `T₃` has no model by boxes, in any dimension, in which
  `A`, `B`, `C` are non-empty; in `ℝ¹` the same holds for all convex (= connected)
  regions.
-/
import GeometricFaithful.Theory

namespace GeometricFaithful

open ContextCalculus.ELCompletion

/-- **One-dimensional Helly.** Three pairwise intersecting intervals have a
common point: the median of the three pairwise witnesses lies in all three. -/
theorem ordConnected_helly2 {s₁ s₂ s₃ : Set ℝ}
    (h₁ : s₁.OrdConnected) (h₂ : s₂.OrdConnected) (h₃ : s₃.OrdConnected)
    (h₁₂ : (s₁ ∩ s₂).Nonempty) (h₂₃ : (s₂ ∩ s₃).Nonempty) (h₁₃ : (s₁ ∩ s₃).Nonempty) :
    (s₁ ∩ s₂ ∩ s₃).Nonempty := by
  obtain ⟨p, hp₁, hp₂⟩ := h₁₂
  obtain ⟨q, hq₂, hq₃⟩ := h₂₃
  obtain ⟨r, hr₁, hr₃⟩ := h₁₃
  rcases le_total p q with hpq | hqp <;> rcases le_total q r with hqr | hrq <;>
    rcases le_total p r with hpr | hrp
  · exact ⟨q, ⟨h₁.out hp₁ hr₁ ⟨hpq, hqr⟩, hq₂⟩, hq₃⟩
  · exact ⟨q, ⟨h₁.out hp₁ hr₁ ⟨hpq, hqr⟩, hq₂⟩, hq₃⟩
  · exact ⟨r, ⟨hr₁, h₂.out hp₂ hq₂ ⟨hpr, hrq⟩⟩, hr₃⟩
  · exact ⟨p, ⟨hp₁, hp₂⟩, h₃.out hr₃ hq₃ ⟨hrp, hpq⟩⟩
  · exact ⟨p, ⟨hp₁, hp₂⟩, h₃.out hq₃ hr₃ ⟨hqp, hpr⟩⟩
  · exact ⟨r, ⟨hr₁, h₂.out hq₂ hp₂ ⟨hqr, hrp⟩⟩, hr₃⟩
  · exact ⟨q, ⟨h₁.out hr₁ hp₁ ⟨hrq, hqp⟩, hq₂⟩, hq₃⟩
  · exact ⟨q, ⟨h₁.out hr₁ hp₁ ⟨hrq, hqp⟩, hq₂⟩, hq₃⟩

/-- An axis-parallel box in `ι → ℝ`: a product of intervals (of any kind). -/
def IsBox {ι : Type} (s : Set (ι → ℝ)) : Prop :=
  ∃ t : ι → Set ℝ, (∀ i, (t i).OrdConnected) ∧ s = Set.pi Set.univ t

/-- **Boxes have Helly number 2** (coordinate-wise one-dimensional Helly). -/
theorem isBox_helly2 {ι : Type} : Helly2 {s : Set (ι → ℝ) | IsBox s} := by
  rintro S₁ ⟨t₁, ht₁, rfl⟩ S₂ ⟨t₂, ht₂, rfl⟩ S₃ ⟨t₃, ht₃, rfl⟩ h₁₂ h₂₃ h₁₃
  obtain ⟨p, hp₁, hp₂⟩ := h₁₂
  obtain ⟨q, hq₂, hq₃⟩ := h₂₃
  obtain ⟨r, hr₁, hr₃⟩ := h₁₃
  rw [Set.mem_univ_pi] at hp₁ hp₂ hq₂ hq₃ hr₁ hr₃
  have hsec : ∀ i, (t₁ i ∩ t₂ i ∩ t₃ i).Nonempty := fun i =>
    ordConnected_helly2 (ht₁ i) (ht₂ i) (ht₃ i)
      ⟨p i, hp₁ i, hp₂ i⟩ ⟨q i, hq₂ i, hq₃ i⟩ ⟨r i, hr₁ i, hr₃ i⟩
  choose m hm using hsec
  exact ⟨m, ⟨Set.mem_univ_pi.2 fun i => (hm i).1.1, Set.mem_univ_pi.2 fun i => (hm i).1.2⟩,
    Set.mem_univ_pi.2 fun i => (hm i).2⟩

/-- **No box model of `T₃` in any dimension.** If `X₁, X₂, X₃` are boxes in
`ι → ℝ` (for any index type `ι`, e.g. `Fin d`), no model of `T₃` has `A`, `B`,
`C` all non-empty. -/
theorem no_box_model {ι : Type} (I : GInterp (ι → ℝ)) (h : models I T3)
    (h₁ : IsBox (region I .X1)) (h₂ : IsBox (region I .X2)) (h₃ : IsBox (region I .X3))
    (hA : (region I .A).Nonempty) (hB : (region I .B).Nonempty)
    (hC : (region I .C).Nonempty) : False :=
  no_nondegenerate_model_of_helly2 isBox_helly2 I h h₁ h₂ h₃ hA hB hC

/-- Closed boxes `[a, b]` are boxes. -/
theorem isBox_Icc {ι : Type} (a b : ι → ℝ) : IsBox (Set.Icc a b) := by
  refine ⟨fun i => Set.Icc (a i) (b i), fun i => Set.ordConnected_Icc, ?_⟩
  ext x
  simp [Set.mem_Icc, Pi.le_def]

/-- **No convex (equivalently, connected) model of `T₃` on the real line.** -/
theorem no_convex_model_dim1 (I : GInterp ℝ) (h : models I T3)
    (h₁ : Convex ℝ (region I .X1)) (h₂ : Convex ℝ (region I .X2)) (h₃ : Convex ℝ (region I .X3))
    (hA : (region I .A).Nonempty) (hB : (region I .B).Nonempty)
    (hC : (region I .C).Nonempty) : False := by
  refine no_nondegenerate_model_of_helly2 (F := {s : Set ℝ | s.OrdConnected}) ?_ I h
    h₁.ordConnected h₂.ordConnected h₃.ordConnected hA hB hC
  intro S₁ hS₁ S₂ hS₂ S₃ hS₃
  exact ordConnected_helly2 hS₁ hS₂ hS₃

end GeometricFaithful
