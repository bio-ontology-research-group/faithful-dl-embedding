import GeometricFaithful

/-! Axiom audit: every main theorem should depend only on the standard axioms
`propext`, `Classical.choice`, `Quot.sound` — in particular, no `sorryAx`. -/

open GeometricFaithful

#print axioms no_nondegenerate_model_of_helly2
#print axioms no_box_model
#print axioms no_convex_model_dim1
#print axioms no_ball_model
#print axioms no_ball_model_euclidean
#print axioms planeModel_models
#print axioms planeModel_nondegenerate
#print axioms planeModel_convex
#print axioms helly_dimension_bound
#print axioms threeElementModel_models
