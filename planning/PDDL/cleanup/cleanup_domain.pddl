(define (domain cleanup)
    (:requirements :strips :typing)
    (:types 
        gripper table tabletop-object - object
        block container - tabletop-object
        mug drawer - container
    ) 
    
    ;; Define predicates
    (:predicates
        (directly-on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (exclusively-occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `exclusively-occupying-gripper` atoms.
        (inside ?tabletop-object - tabletop-object ?container - container) ; whether the object is inside the container.
        (large-enough-for-gripper-to-reach-inside ?container - container ?gripper - gripper) ; whether the gripper can reach inside the container.
        (open ?container - container) ; whether the container is open enough to fit the object through the opening
        (small-enough-for-gripper-to-pick-up ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object can be picked up.
        (small-enough-to-fit-in-container ?tabletop-object - tabletop-object ?container - container) ; whether the object can fit inside the container.
        
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


    (:action free-gripper-from-large-object
     :parameters (?tabletop-object - tabletop-object ?gripper - gripper)
     :precondition (and (not (small-enough-for-gripper-to-pick-up ?tabletop-object ?gripper)) (exclusively-occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
     :effect (and
        (not (exclusively-occupying-gripper ?tabletop-object ?gripper))
        (free ?gripper))
    )


    (:action open-drawer
     :parameters (?drawer - drawer ?gripper - gripper)
     :precondition (and (not (open ?drawer)) (free ?gripper))
     :effect (and
        (exclusively-occupying-gripper ?drawer ?gripper)
        (not (free ?gripper))
        (open ?drawer))
    )


    (:action close-drawer
     :parameters (?drawer - drawer ?gripper - gripper)
     :precondition (and (open ?drawer) (free ?gripper))
     :effect (and
        (exclusively-occupying-gripper ?drawer ?gripper)
        (not (free ?gripper))
        (not (open ?drawer)))
    )


    (:action place-in-drawer-from-gripper
     :parameters (?drawer - drawer ?tabletop-object - tabletop-object ?gripper - gripper)
     :precondition (and (open ?drawer) (small-enough-to-fit-in-container ?tabletop-object ?drawer) (large-enough-for-gripper-to-reach-inside ?drawer ?gripper) (exclusively-occupying-gripper ?tabletop-object ?gripper) (not (free ?gripper)))
     :effect (and
        (inside ?tabletop-object ?drawer)
        (not (exclusively-occupying-gripper ?tabletop-object ?gripper))
        (free ?gripper))
    )
    
)