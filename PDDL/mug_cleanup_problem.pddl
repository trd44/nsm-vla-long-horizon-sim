(define (problem mug-cleanup)
  (:domain tabletop-manipulation-domain)
  (:objects 
	mug1 - holdable
	drawer1 - drawer
	nothing - holdable
  )
  (:init 
	(mug mug1)
	(drawer drawer1)
	(not (open drawer1))
	(holding nothing)
	(not (in-drawer mug1))
  )
  (:goal 
	(and 
	  (in-drawer mug1)
	  (holding nothing)
	)
  )
)