(define (problem hanoi)
  (:domain hanoi)
  (:objects 
    cube0 cube1 cube2 cube3 cube4 cube5 - disk
    platform1 platform2 table - peg
  )
  (:init 
(clear platform2 )
(clear platform1 )
(clear cube3 )
(clear cube2 )
(clear cube1 )
(clear cube0 )
    (free-gripper)
    (on cube0 table)
    (on cube1 table)
    (on cube2 table)
    (on cube3 table)
    (on cube4 table)
    (on cube5 table)
  )
  (:goal 
    (and
      (on cube3 platform1)
      (on cube2 platform2)
      (on cube1 platform2)
      (on cube0 platform2)
  )
    )
)
