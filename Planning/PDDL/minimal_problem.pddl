(define (problem minimal-problem)
    (:domain minimal)
    (:objects
        table1 - table
        mug1 - mug
        coffee-pod1 - coffee-pod
    )
    (:init
        (clear table1)
    )
    (:goal
        (on mug1 table1)
    )
)
