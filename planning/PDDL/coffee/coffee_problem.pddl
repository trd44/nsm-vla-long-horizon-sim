(define (problem coffee)
  (:domain coffee)
  (:objects
    lid1 - coffee-machine-lid
    drawer1 - drawer
    coffee-pod1 - coffee-pod
    coffee-pod-holder1 - coffee-pod-holder
    table1 - table
    mug1 - mug
    gripper1 - gripper
  )

  ;initial symbolic state of the task using ONLY available predicates
  (:init
    (attached lid1 coffee-pod-holder1)
    (can-flip-up lid1)
    (directly-on-table mug1 table1)
    (directly-on-table drawer1 table1)
    (free gripper1)
    (in coffee-pod1 drawer1)
    (open mug1)
    (small-enough-for-gripper-to-pick-up coffee-pod1 gripper1)
    (small-enough-for-gripper-to-pick-up mug1 gripper1)
    (small-enough-for-gripper-to-pick-up coffee-pod-holder1 gripper1)
    (under mug1 coffee-pod-holder1)
    (upright mug1)
  )

  (:goal 
    (and
      (in coffee-pod1 coffee-pod-holder1)
      (under mug1 coffee-pod-holder1)
      (upright mug1)
    )
  )
)