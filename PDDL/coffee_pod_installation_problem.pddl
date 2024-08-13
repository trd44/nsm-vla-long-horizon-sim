(define (problem coffee-pod-installation)
  (:domain tabletop-manipulation-domain)
  (:objects 
	coffee-pod1 - coffee-pod
	coffee-machine1 - coffee-machine
	nothing - nothing
  )
  (:init 
	(holding nothing)
	(not (open coffee-machine1))
	(not (in-drawer coffee-pod1))
  )
  (:goal 
	(and 
	  (pod-installed coffee-pod1)
	  (not (open coffee-machine1))
	)
  )
)