(define (problem generic)
  (:domain generic)
  (:objects
    drawer1 - drawer
    coffee-pod1 - coffee-pod
    coffee-dispenser1 - coffee-dispenser
    coffee-dispenser-lid1 - coffee-dispenser-lid
    table1 - table
    mug1 - mug
    gripper1 - gripper
  )
  (:init
    (free gripper1)
	(not (open coffee-dispenser1))
	(on coffee-pod1 table1)
	(on mug1 table1)
    ; (on mug1 table1)
    ; (free gripper1)
    ; (on coffee-pod1 table1)
    ; (not (open coffee-dispenser1))
  )
  (:goal
    (and
      (in coffee-pod1 coffee-dispenser1)
      (not (open coffee-dispenser1))
      (under mug1 coffee-dispenser1)
    )
  )
)