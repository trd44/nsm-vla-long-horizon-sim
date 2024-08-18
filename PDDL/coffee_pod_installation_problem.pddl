(define (problem coffee-pod-installation)
  (:domain coffee-pod-installation)
  (:objects 
	coffee-pod - coffee-pod
	coffee-machine - coffee-machine
  	table - table
	gripper - gripper
  )
  (:init 
	(free gripper)
	(not (open coffee-machine1))
	(on coffee-pod table)
  )
  (:goal 
	(and 
	  (in coffee-pod coffee-machine)
	  (not (open coffee-machine))
	)
  )
)