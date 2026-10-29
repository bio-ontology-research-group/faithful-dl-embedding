/-
  GeometricFaithful/Theory.lean
  =============================
  The theory

      A ≡ X₁ ⊓ X₂,  B ≡ X₂ ⊓ X₃,  C ≡ X₁ ⊓ X₃,
      A ⊓ B ⊑ ⊥,    B ⊓ C ⊑ ⊥,    A ⊓ C ⊑ ⊥

  written in the EL++ normal form of the kobayashi-marust certification library
  (`ContextCalculus.ELCompletion`: `nf1 sub sup` is `sub ⊑ sup`, `nf2 l r sup` is
  `l ⊓ r ⊑ sup`), its interpretations (`Interp D CN RN top bot`, with `concept c`
  the extension of `c` as a predicate on the domain `D`), and the generic
  Helly-type obstruction: if the three regions of X₁, X₂, X₃ come from a family in
  which pairwise intersecting triples always have a common point, then the
  regions of A, B, C cannot all be non-empty.
-/
import Mathlib
import ContextCalculus.ELCompletion

namespace GeometricFaithful

open ContextCalculus.ELCompletion

/-- Concept names of the theory, together with ⊤ and ⊥. -/
inductive CN where
  | X1 | X2 | X3 | A | B | C | top | bot
  deriving DecidableEq, Repr

/-- No role names are needed. -/
abbrev RN := Empty

open CN in
/-- The theory `T₃` in EL normal form. -/
def T3 : Ontology CN RN :=
  [ .nf1 A X1, .nf1 A X2, .nf2 X1 X2 A,
    .nf1 B X2, .nf1 B X3, .nf2 X2 X3 B,
    .nf1 C X1, .nf1 C X3, .nf2 X1 X3 C,
    .nf2 A B bot, .nf2 B C bot, .nf2 A C bot ]

variable {D : Type}

/-- A geometric interpretation: a DL interpretation whose domain is a space `D`. -/
abbrev GInterp (D : Type) := Interp D CN RN CN.top CN.bot

/-- The region (extension) of a concept name. -/
def region (I : GInterp D) (c : CN) : Set D := {x | I.concept c x}

theorem mem_region {I : GInterp D} {c : CN} {x : D} : x ∈ region I c ↔ I.concept c x := Iff.rfl

section Regions

variable {I : GInterp D} (h : models I T3)
include h

theorem region_A : region I .A = region I .X1 ∩ region I .X2 := by
  ext x; constructor
  · intro hx
    exact ⟨h (.nf1 .A .X1) (by simp [T3]) x hx, h (.nf1 .A .X2) (by simp [T3]) x hx⟩
  · rintro ⟨h1, h2⟩
    exact h (.nf2 .X1 .X2 .A) (by simp [T3]) x h1 h2

theorem region_B : region I .B = region I .X2 ∩ region I .X3 := by
  ext x; constructor
  · intro hx
    exact ⟨h (.nf1 .B .X2) (by simp [T3]) x hx, h (.nf1 .B .X3) (by simp [T3]) x hx⟩
  · rintro ⟨h1, h2⟩
    exact h (.nf2 .X2 .X3 .B) (by simp [T3]) x h1 h2

theorem region_C : region I .C = region I .X1 ∩ region I .X3 := by
  ext x; constructor
  · intro hx
    exact ⟨h (.nf1 .C .X1) (by simp [T3]) x hx, h (.nf1 .C .X3) (by simp [T3]) x hx⟩
  · rintro ⟨h1, h2⟩
    exact h (.nf2 .X1 .X3 .C) (by simp [T3]) x h1 h2

theorem region_A_inter_B : region I .A ∩ region I .B = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro ha hb
  exact I.bottom_false x (h (.nf2 .A .B .bot) (by simp [T3]) x ha hb)

/-- The triple intersection of the `Xᵢ` is the region of `A ⊓ B`, hence empty. -/
theorem triple_inter_empty : region I .X1 ∩ region I .X2 ∩ region I .X3 = ∅ := by
  have hAB := region_A_inter_B h
  rw [region_A h, region_B h] at hAB
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  rintro ⟨h1, h2⟩ h3
  have : x ∈ (region I .X1 ∩ region I .X2) ∩ (region I .X2 ∩ region I .X3) := ⟨⟨h1, h2⟩, ⟨h2, h3⟩⟩
  rw [hAB] at this
  exact this

end Regions

/-- A family of sets in which any three pairwise intersecting members have a
common point ("Helly number at most 2", for triples). -/
def Helly2 (F : Set (Set D)) : Prop :=
  ∀ S₁ ∈ F, ∀ S₂ ∈ F, ∀ S₃ ∈ F,
    (S₁ ∩ S₂).Nonempty → (S₂ ∩ S₃).Nonempty → (S₁ ∩ S₃).Nonempty →
      (S₁ ∩ S₂ ∩ S₃).Nonempty

/-- **Helly-type obstruction.** If `X₁, X₂, X₃` are interpreted by members of a
family with the Helly property for triples, no model of `T₃` can make `A`, `B`
and `C` all non-empty, whatever shapes `A`, `B`, `C` themselves have. -/
theorem no_nondegenerate_model_of_helly2 {F : Set (Set D)} (hF : Helly2 F)
    (I : GInterp D) (h : models I T3)
    (h1 : region I .X1 ∈ F) (h2 : region I .X2 ∈ F) (h3 : region I .X3 ∈ F)
    (hA : (region I .A).Nonempty) (hB : (region I .B).Nonempty)
    (hC : (region I .C).Nonempty) : False := by
  rw [region_A h] at hA
  rw [region_B h] at hB
  rw [region_C h] at hC
  have := hF _ h1 _ h2 _ h3 hA hB hC
  rw [triple_inter_empty h] at this
  exact Set.not_nonempty_empty this

/-- The theory is consistent and every name is satisfiable: the three-element
model `{0,1,2}` with `Xⱼ = {i | i ≠ j}`. -/
def threeElementModel : GInterp (Fin 3) where
  concept
    | .X1 => fun i => i ≠ 0
    | .X2 => fun i => i ≠ 1
    | .X3 => fun i => i ≠ 2
    | .A => fun i => i = 2
    | .B => fun i => i = 0
    | .C => fun i => i = 1
    | .top => fun _ => True
    | .bot => fun _ => False
  role := fun r => r.elim
  top_true := fun _ => trivial
  bottom_false := fun _ h => h

theorem threeElementModel_models : models threeElementModel T3 := by
  intro c hc
  simp only [T3, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    (simp only [satClause, threeElementModel]; decide)

theorem threeElementModel_nondegenerate :
    (region threeElementModel .A).Nonempty ∧ (region threeElementModel .B).Nonempty ∧
    (region threeElementModel .C).Nonempty :=
  ⟨⟨2, rfl⟩, ⟨0, rfl⟩, ⟨1, rfl⟩⟩

end GeometricFaithful
