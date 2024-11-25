

(define (domain coffee)
    (:requirements :strips :typing)
    (:types
        gripper table tabletop-object - object
        coffee-pod coffee-machine-lid container - tabletop-object
        mug coffee-pod-holder drawer - container
    )

    ;; Define predicates
    (:predicates
        (attached ?lid - coffee-machine-lid ?holder - coffee-pod-holder)
        (can-flip-up ?lid - coffee-machine-lid)
        (can-flip-down ?lid - coffee-machine-lid)
        (directly-on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (exclusively-occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false. 
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `exclusively-occupying-gripper` atoms.
        (in ?tabletop-object - tabletop-object ?container - container) ; whether the object is inside the container.
        (open ?container - container) ; whether the container is open
        (small-enough-for-gripper-to-pick-up ?tabletop-object - tabletop-object ?gripper - gripper); whether the object can be picked up.(inside ?tabletop-object - tabletop-object ?container - container) ; whether the object is inside the container.
        (under ?mug - mug ?holder - coffee-pod-holder) ; whether the bottom tabletop object is under the top tabletop object.
        (upright ?mug - mug) ; whether the mug is upright.
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-from-tabletop
     :parameters (?tabletop-object - tabletop-object ?table - table ?gripper - gripper)
     :precondition (and (directly-on-table ?tabletop-object ?table) (small-enough-for-gripper-to-pick-up ?tabletop-object ?gripper) (free ?gripper))
     :effect (and
        (exclusively-occupying-gripper ?tabletop-object ?gripper)
        (not (free ?gripper))
        (not (directly-on-table ?tabletop-object ?table)))
    )


    (:action open-coffee-pod-holder
     :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
     :precondition (and (not (open ?holder)) (can-flip-up ?lid) (attached ?lid ?holder) (free ?gripper))
     :effect (and
        (exclusively-occupying-gripper ?lid ?gripper)
        (not (free ?gripper))
        (open ?holder)
        (can-flip-down ?lid))
    )


    (:action close-coffee-pod-holder
     :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
     :precondition (and (open ?holder) (can-flip-down ?lid) (attached ?lid ?holder) (free ?gripper))
     :effect (and
        (exclusively-occupying-gripper ?lid ?gripper)
        (not (free ?gripper))
        (not (open ?holder)))
    )


    (:action free-gripper-from-large-object
     :parameters (?tabletop-object - tabletop-object ?gripper - gripper)
     :precondition (and (not (small-enough-for-gripper-to-pick-up ?tabletop-object ?gripper)) (exclusively-occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
     :effect (and
        (not (exclusively-occupying-gripper ?tabletop-object ?gripper))
        (free ?gripper))
    )


    (:action place-pod-in-holder-from-gripper
     :parameters (?pod - coffee-pod ?holder - coffee-pod-holder ?gripper - gripper)
     :precondition (and (exclusively-occupying-gripper ?pod ?gripper) (open ?holder) (not (free ?gripper)))
     :effect (and
        (not (exclusively-occupying-gripper ?pod ?gripper))
        (free ?gripper)
        (in ?pod ?holder))
    )


    (:action place-mug-under-holder-from-gripper
     :parameters (?mug - mug ?holder - coffee-pod-holder ?gripper - gripper)
     :precondition (and (exclusively-occupying-gripper ?mug ?gripper) (not (free ?gripper)))
     :effect (and
        (under ?mug ?holder)
        (upright ?mug)
        (not (exclusively-occupying-gripper ?mug ?gripper))
        (free ?gripper))
    )

)