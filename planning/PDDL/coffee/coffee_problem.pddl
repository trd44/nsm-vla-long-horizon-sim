(define (problem coffee)
  (:domain coffee)
  (:objects
    drawer1 - drawer
    coffee-pod1 - coffee-pod
    coffee-pod-holder1 - coffee-pod-holder
    table1 - table
    mug1 - mug
    gripper1 - gripper
  )

  ;initial symbolic state of the task using ONLY available predicates
  (:init
    (closed coffee-pod-holder1)
    (closed drawer1)
    (directly-on-table mug1 table1)
    (directly-on-table drawer1 table1)
    (free gripper1)
    (inside coffee-pod1 drawer1)
    (open-enough-to-fit-through mug1 coffee-pod1)
    (small-enough-for-gripper-to-pick-up coffee-pod1 gripper1)
    (small-enough-for-gripper-to-pick-up mug1 gripper1)
    (upright mug1)
  )

  (:goal 
    (and
      (inside coffee-pod1 coffee-pod-holder1)
      (closed coffee-pod-holder1)
      (under mug1 coffee-pod-holder1)
      (upright mug1)
    )
  )
)