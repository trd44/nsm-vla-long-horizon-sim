(define (problem cleanup_problem)
    (:domain cleanup)

    (:objects
        block1 - block
        mug1 - mug
        drawer1 - drawer
        table1 - table
        gripper1 - gripper
    )

    (:init
        (free gripper1)
        (small-enough-to-pick-up mug1)
        (small-enough-to-pick-up block1)
        (large-enough-for-gripper-to-reach-inside drawer1)
        (open mug1)
        (in block1 mug1)
        (directly-on-table mug1 table1)
        (directly-on-table drawer1 table1)
    )

    (:goal
        (and
            (in block1 drawer1)
            (not (open drawer1))
        )
    )
)