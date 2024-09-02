(define (problem tabletop)
  (:domain tabletop)
  (:objects
    drawer1 - container
    coffee-pod1 - coffee-pod
    coffee-pod-holder1 - coffee-pod-holder
    coffee-machine-lid1 - coffee-machine-lid
    table1 - table
    mug1 - mug
    gripper1 - gripper
  )

  ;initial symbolic state of the task using ONLY available predicates
  (:init
    (free gripper1)
    (on coffee-pod1 table1)
    (under mug1 coffee-pod-holder1)
  )

  (:goal 
    (and
      (in coffee-pod1 coffee-pod-holder1)
      (under mug1 coffee-pod-holder1)
    )
  )
)