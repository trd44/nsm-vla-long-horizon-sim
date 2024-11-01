(define (domain nut_assembly)
    (:requirements :strips :typing)
    (:types 
        gripper table tabletop-object - object
        nut peg - tabletop-object
        round-peg square-peg - peg
        round-nut square-nut - nut
    ) 
    
    ;; Define predicates
    (:predicates
        (small-enough-for-gripper-to-pick-up ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object can be picked up.
        (directly-on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (on-peg ?nut - nut ?peg - peg) ; whether the nut is on the peg.
        (exclusively-occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (shapes-match ?nut - nut ?peg - peg) ; whether the nut matches the peg.
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `exclusively-occupying-gripper` atoms.
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-nut-from-tabletop
        :parameters (?nut - nut ?table - table ?gripper - gripper) 
        :precondition (and (directly-on-table ?nut ?table) (small-enough-for-gripper-to-pick-up ?nut ?gripper) (free ?gripper)) 
        :effect (and (exclusively-occupying-gripper ?nut ?gripper) (not (directly-on-table ?nut ?table)) (not (free ?gripper)))
    ) 
    

    (:action put-nut-on-peg
        :parameters (?nut - nut ?peg - peg ?gripper - gripper)
        :precondition (and (shapes-match ?nut ?peg) (exclusively-occupying-gripper ?nut ?gripper) (not (free ?gripper)))
        :effect (and (on-peg ?nut ?peg) (not (exclusively-occupying-gripper ?nut ?gripper)) (free ?gripper))
    )
    
)