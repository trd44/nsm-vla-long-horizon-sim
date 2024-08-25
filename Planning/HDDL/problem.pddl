(define (problem coffee-prep-problem)
	(:domain coffee-prep)
	
	;; Define objects
	(:objects
		mug1 - mug
		pod1 - coffee-pod
		dispenser1 - coffee-dispenser
		lid1 - coffee-dispenser-lid
		table1 - table
		gripper1 - gripper
	)
	
	;; Define initial state
	(:init
		(on pod1 table1)
		(free gripper1)
		(not (open dispenser1))
	)
	
	;; Define tasks
	(:tasks
		(prepare-coffee pod1 dispenser1 lid1 table1 gripper1)
	)
)