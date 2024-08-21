(define (problem coffee-pod-installation)
  (:domain generic)
  (:objects 
	coffee-pod1 - coffee-pod
	coffee-dispenser1 - coffee-dispenser
  	table1 - table
	gripper1 - gripper
  	mug1 - mug
  	coffee-dispenser-lid1 - coffee-dispenser-lid
  )
  (:init 
	(free gripper1)
	(not (open coffee-dispenser1))
	(on coffee-pod1 table1)
	(under mug1 coffee-dispenser1)
  )
  (:goal 
	(and 
	  (in coffee-pod1 coffee-dispenser1)
	  (not (open coffee-dispenser1))
	  (under mug1 coffee-dispenser1)
	)
  )
)