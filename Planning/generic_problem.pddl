(define (problem coffee-pod-installation-problem)
    (:domain coffee-pod-installation)
    (:objects
        drawer - container
        table dispenser - support
        gripper - gripper
        pod - coffee-pod
        cup - mug
        coffee-machine - container
    )
    (:init
        (not (open drawer))
        (in pod drawer)
        (on cup table)
        (free gripper)
        (not (open coffee-machine))
    )
    (:goal (and
        (in pod coffee-machine)
        (on cup dispenser)
    ))
)
