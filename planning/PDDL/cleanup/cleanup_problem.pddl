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
        (directly-on-table mug1 table1)
        (directly-on-table drawer1 table1)
        (free gripper1)
        (inside block1 mug1)
        (large-enough-for-gripper-to-reach-inside drawer1 gripper1)
        (open mug1)
        (small-enough-for-gripper-to-pick-up mug1 gripper1)
        (small-enough-for-gripper-to-pick-up block1 gripper1)
        (small-enough-to-fit-in-container block1 drawer1)
        (small-enough-to-fit-in-container block1 mug1)
        (small-enough-to-fit-in-container mug1 drawer1)
    )

    (:goal
        (and
            (inside block1 drawer1)
            (not (inside mug1 drawer1))
            (not (open drawer1))
        )
    )
)