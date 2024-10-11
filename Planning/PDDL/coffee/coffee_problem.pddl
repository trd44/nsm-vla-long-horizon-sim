(define (problem coffee)
  (:domain coffee)
  (:objects
    drawer1 - drawer
    coffee-pod1 - coffee-pod
    coffee-pod-holder1 - coffee-pod-holder
    coffee-machine-lid1 - coffee-machine-lid
    table1 - table
    mug1 - mug
    gripper1 - gripper
  )

  ;initial symbolic state of the task using ONLY available predicates
  (:init
    (attached coffee-machine-lid1 coffee-pod-holder1)
    (can-flip-up coffee-machine-lid1)
    (not-attached-and-small-enough-to-pick-up coffee-pod1)
    (not-attached-and-small-enough-to-pick-up mug1)
    (free gripper1)
    (in coffee-pod1 drawer1)
    (open mug1)
    (on-table mug1 table1)
    (on-table drawer1 table1)
  )

  (:goal 
    (and
      (in coffee-pod1 coffee-pod-holder1)
      (not (open coffee-pod-holder1))
      (under mug1 coffee-pod-holder1)
    )
  )
)